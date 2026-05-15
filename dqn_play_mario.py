import retro
import numpy as np
import cv2
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from torch.distributions.categorical import Categorical

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class UpscaleRender(gym.Wrapper):
    def __init__(self, env, scale: int = 4, interpolation=cv2.INTER_NEAREST):
        super().__init__(env)
        self.scale = scale
        self.interpolation = interpolation

    def render(self):
        frame = self.env.render()
        if frame is not None and isinstance(frame, np.ndarray):
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (w * self.scale, h * self.scale),
                interpolation=self.interpolation
            )
        return frame

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)

        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
     torch.nn.init.orthogonal_(layer.weight, std)
     torch.nn.init.constant_(layer.bias, bias_const)
     return layer

class DQN(nn.Module):
    def __init__(self, num_actions, in_channels=3):
        super (DQN, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),  # [84x84 -> 20x20]
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), # [20x20 -> 9x9]
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), # [9x9 -> 7x7]
            nn.ReLU(),
            nn.Flatten(),
            NoisyLinear(64 * 7 * 7, 512),
            nn.ReLU()
        )

        self.val = NoisyLinear(512, 1)
        self.advantage = NoisyLinear(512, num_actions)

    def features(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)# adds batch dimension at index 0
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        x = self.network(x)
        return x
    
    def get_qvals(self, x):
        x = self.features(x)
        advantage = self.advantage(x)
        x = self.val(x) + advantage - advantage.mean(dim=1, keepdim=True)
        return x
    
    def forward(self, x):
        x = self.features(x)
        advantage = self.advantage(x)
        x = self.val(x) + advantage - advantage.mean(dim=1, keepdim=True)
        return x
    
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
    
class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def add(self, x):
        self.buffer.append(x)

def stack_obs(frame_stack):
    return torch.stack(list(frame_stack), dim=-1)

def selection_to_action(actionSelection):
        if(actionSelection == 0):
            #jump right fast
            actionSelection = [1,0,0,0,0,0,0,1,1]
        elif(actionSelection == 1):
            #jump
            actionSelection = [0,0,0,0,0,0,0,0,1]
        elif(actionSelection == 2):
            #right fast
            actionSelection = [1,0,0,0,0,0,0,1,0]
        elif(actionSelection == 3):
            #right
            actionSelection = [0,0,0,0,0,0,0,1,0]
        elif(actionSelection == 4):
            #left
            actionSelection = [0,0,0,0,0,0,1,0,0]
        elif(actionSelection == 5):
            #jump right
            actionSelection = [0,0,0,0,0,0,0,1,1]
        elif(actionSelection == 6):
            #stay still
            actionSelection = [0,0,0,0,0,0,0,0,0]
        return actionSelection

if __name__ == "__main__":
    run_name = f"run_{int(time.time())}"

    seed = 1

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_env = retro.make(game='SuperMarioBros3-Nes', render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(base_env)
    env = UpscaleRender(base_env, scale=4)   # 256×240 → 1024×960
    env = gym.wrappers.RecordVideo(env, video_folder="dqn_mario_play_vid", episode_trigger=lambda t: t % 1 == 0, fps=60)

    obs, _ = env.reset()
    obs = torch.from_numpy(obs).float().to(device)

    print(obs.shape)

    k=4

    num_actions = 7

    agent = DQN(num_actions, k).to(device)
    agent.load_state_dict(torch.load("pretrained_model/agent_model_5100000", map_location=device))



    #lr=1e-4
    lr=7e-5 #best?

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps = 1e-5)

    num_steps = 20000000
    batch_size = 32
    gamma = 0.99
    target_update_freq = 2000
    episodic_return = 0
    episode_length = 0
    frame_skip = 4

    obs, _ = env.reset()
    stagnant = 0
    obs = cv2.cvtColor(cv2.resize(obs, (84, 84)), cv2.COLOR_BGR2GRAY)
    obs = torch.tensor(obs, dtype=torch.uint8).to("cpu")

    k = 4
    frame_stack = deque(maxlen=k)
    for _ in range(k):
        frame_stack.append(obs)
    obs_stacked = stack_obs(frame_stack)  # (H, W, 3*k)

    buffer = ReplayBuffer(maxlen=250000)
    ram = env.unwrapped.get_ram()
    x_hi = int(ram[0x0075])
    x_lo = int(ram[0x0090])
    maxX = x_hi*256 + x_lo


    for step in range(1, num_steps):

        agent.reset_noise()
        with torch.no_grad():
            qvals = agent.get_qvals(obs_stacked.to(device).float())
            action = qvals.argmax(dim=-1).item()

        action_selection = selection_to_action(action)

        reward = 0
        for i in range(frame_skip):
            next_obs, _, terminated, truncated, info = env.step(action_selection)
            ns_recolor = cv2.cvtColor(next_obs, cv2.COLOR_RGB2BGR)
            resized_frame = cv2.resize(ns_recolor, (next_obs.shape[1]*2, next_obs.shape[0]*2), interpolation=cv2.INTER_AREA)
            cv2.imshow('Super Mario Bros 3', resized_frame)
            key = cv2.waitKey(1) & 0xFF   # <- REQUIRED for refresh
            if key == ord('q'):
                done = True
                break

            #~~~~~~~~~~~~~calculate reward~~~~~~~~~~~~~#
            ram = env.unwrapped.get_ram()
            won = ram[0x000D6]
            died = ram[0x000F1]
            far = int(ram[0x00023])
            x_hi = int(ram[0x0075])
            x_lo = int(ram[0x0090])

            r = 0

            r-= 0.0025

            end_reward = 0
            if won == 216:
                terminated = True
                end_reward += 5.0
            elif died > 0:
                terminated = True
                end_reward -= 2.0

            global_x = x_hi*256 + x_lo
            delta = global_x - maxX
            if delta > 0:
                r += 0.01 * delta
                maxX = global_x
                stagnant = 0
            else:
                stagnant += 1

            if far >= 176:  #encourage hitting the goal instead of running past
                r = 0
            if stagnant > 180:
                truncated = True

            reward += r
            reward += end_reward
            #~~~~~~~~~~~~~calculate reward~~~~~~~~~~~~~#

            done = terminated | truncated


            next_obs = cv2.cvtColor(cv2.resize(next_obs, (84, 84)), cv2.COLOR_BGR2GRAY)
            next_obs = torch.tensor(next_obs, dtype=torch.uint8).to("cpu")
            frame_stack.append(next_obs)
            next_obs_stacked = stack_obs(frame_stack)

            if done:
                break

        #store (state, action, reward, next_state, done) transitions
        #buffer.add([obs_stacked, action, reward, next_obs_stacked, done])

        episodic_return+=reward
        episode_length+=1

        if done:
            next_obs, _ = env.reset()
            ram = env.unwrapped.get_ram()
            x_hi = int(ram[0x0075])
            x_lo = int(ram[0x0090])
            maxX = x_hi*256 + x_lo
            stagnant = 0
            next_obs = cv2.cvtColor(cv2.resize(next_obs, (84, 84)), cv2.COLOR_BGR2GRAY)
            next_obs = torch.tensor(next_obs, dtype=torch.uint8).to("cpu")

            frame_stack.clear()
            for _ in range(k):
                frame_stack.append(next_obs)
            next_obs_stacked = stack_obs(frame_stack)

            print(f"[Episode End] Step {step} | Return={episodic_return:.2f} | Len={episode_length}")
            episodic_return = 0
            episode_length = 0

        obs_stacked = next_obs_stacked



    env.close()
    cv2.destroyAllWindows()