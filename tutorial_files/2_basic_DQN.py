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

'''
~~~~~~~~~~~~~~~~~~~~~~~~~
TUTORIAL PART 2 README
~~~~~~~~~~~~~~~~~~~~~~~~~

This file trains a basic Deep Q-Network (DQN) agent to play Super Mario Bros. 3.

The goal of DQN is to learn a Q-function:

    Q(state, action) -> expected future reward

We use a neural network to represent Q. The network looks at
the last few grayscale game frames and predicts one Q-value for each allowed action.

This file includes the main DQN ingredients from the video:

- A convolutional neural network Q-function
- Epsilon-greedy exploration (random action selection)
- A replay buffer
- Discounted future reward with gamma
- A target network
- TD target calculation
- TD loss
- Frame preprocessing: RGB -> grayscale -> 84x84
- Frame stacking
- Frame skipping
- TensorBoard logging
- Periodic gameplay video recording
'''

#Uses GPU if available (cuda)
#Otherwise uses CPU  
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
     """
     Initialize a neural network layer.

     Orthogonal initialization is commonly used in RL because it can make early
     training more stable than completely default random initialization.
     """
     torch.nn.init.orthogonal_(layer.weight, std)
     torch.nn.init.constant_(layer.bias, bias_const)
     return layer

#DQN Mnih et al. 2013 https://arxiv.org/pdf/1312.5602
#
# Input shape:
#   (batch, 84, 84, 4)
#
# Output shape:
#   (batch, num_actions)
#
class DQN(nn.Module):
    def __init__(self, num_actions, in_channels=3):
        super (DQN, self).__init__()


        # Convolutional layers turn the stack of game frames into a compact feature vector.
        # The comments show how the image size changes after each convolution.
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),  # [84x84 -> 20x20]
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), # [20x20 -> 9x9]
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), # [9x9 -> 7x7]
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU()
        )

        # Final layer maps the 512 learned features to one Q-value per action.
        self.actions = layer_init(nn.Linear(512, num_actions))

    def features(self, x):
        # If a single state is passed in with no batch dimension, add one.
                # Example: (84, 84, 4) -> (1, 84, 84, 4)
        if x.dim() == 3:
            x = x.unsqueeze(0)# adds batch dimension at index 0

        # PyTorch convolution layers expect channels first:
        # (batch, height, width, channels) -> (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        x = self.network(x)
        return x
    
    def get_qvals(self, x):
        # Convenience function used during action selection.
        x = self.features(x)
        x = self.actions(x)
        return x
    
    def forward(self, x):
        # Standard PyTorch forward pass.
        # Returns Q-values for every possible action.
        x = self.features(x)
        x = self.actions(x)
        return x
    
    
class ReplayBuffer():
    def __init__(self, maxlen):
        # The replay buffer stores past experience so the agent can learn from
        # randomized batches instead of only the most recent transition.
        self.buffer = deque(maxlen=maxlen)

    def sample(self, batch_size):
        # Random sampling breaks up correlations between consecutive frames.
        return random.sample(self.buffer, batch_size)
    
    def add(self, x):
        # Each item is one transition:
        # (state, action, reward, next_state, done)
        self.buffer.append(x)

def stack_obs(frame_stack):
    # Combine the most recent k grayscale frames into one state.
    # This lets the agent infer motion, such as whether Mario is moving or falling.
    return torch.stack(list(frame_stack), dim=-1)

def selection_to_action(actionSelection):
        """
        Convert a small discrete action index into the 9-button NES action vector.

        The DQN only chooses among a simplified set of useful Mario actions instead
        of every possible controller button combination.
        """
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
    # Give each run a unique name so TensorBoard logs do not overwrite each other.
    run_name = f"run_{int(time.time())}"

    writer = SummaryWriter(f"base_dqn_mario_runs/{run_name}")

    # Set random seeds to make results more reproducible.
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the Super Mario Bros. 3 environment.
    # render_mode="rgb_array" is required for video recording and image preprocessing.
    base_env = retro.make(game='SuperMarioBros3-Nes', render_mode="rgb_array")

    # Record episode statistics such as episode length and return.
    env = gym.wrappers.RecordEpisodeStatistics(base_env)

    # Save gameplay videos periodically so we can visually inspect training progress.
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="base_dqn_mario_vid",
        episode_trigger=lambda t: t % 1000 == 0,
        fps=60
    )

    obs, _ = env.reset()
    obs = torch.from_numpy(obs).float().to(device)

    print(obs.shape) # Raw image shape before preprocessing.

    # Number of frames stacked into one state.
    # A single frame cannot show velocity, so we stack multiple frames to reveal motion.
    k=4

    # Number of simplified actions the agent can choose from.
    num_actions = 7


    # The online network is the network we train every update step.
    agent = DQN(num_actions, k).to(device)

    # The target network is a slower-moving copy used to compute the TD target.
    # This makes training more stable because the target does not change every gradient step.
    target = DQN(num_actions, k).to(device)
    target.load_state_dict(agent.state_dict())

    lr=7e-5 #best?

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps = 1e-5)

    # Main training hyperparameters.
    num_steps = 10000000
    batch_size = 32
    gamma = 0.99
    target_update_freq = 2000
    episodic_return = 0
    episode_length = 0

    # Repeat each selected action for multiple emulator frames.
    # This speeds up training and makes actions like jumping last longer.
    frame_skip = 4


    # Reset the environment and preprocess the first observation.
    obs, _ = env.reset()

    #reset game if this gets too high without progress
    stagnant = 0


    # Convert the RGB frame to a smaller grayscale frame.
    # This keeps the important visual information while reducing computation.
    obs = cv2.cvtColor(cv2.resize(obs, (84, 84)), cv2.COLOR_BGR2GRAY)
    obs = torch.tensor(obs, dtype=torch.uint8).to("cpu")


    # Initialize the frame stack by repeating the first frame k times.
    # This gives the first state the expected shape before we have k real frames.
    k = 4
    frame_stack = deque(maxlen=k)
    for _ in range(k):
        frame_stack.append(obs)
    obs_stacked = stack_obs(frame_stack) 


    # Store transitions for replay.
    buffer = ReplayBuffer(maxlen=250000)

    # Read Mario's initial global x-position from RAM.
    # x_hi and x_lo are combined because the level position spans more than one byte.
    ram = env.unwrapped.get_ram()
    x_hi = int(ram[0x0075])
    x_lo = int(ram[0x0090])
    maxX = x_hi*256 + x_lo

    # Epsilon-greedy exploration settings.
    # The agent starts mostly random, then gradually relies more on its learned Q-values.
    epsilon_start = 1
    epsilon_end = 0.05
    decay_steps = 500000

    for step in range(1, num_steps):

        # Save model checkpoints every so often.
        if step % 300000 == 0:
            torch.save(agent.state_dict(), f"agent_model_{step}")
            #torch.save(target.state_dict(), f"dqn_video_models/target_model_{step}")

        # Get Q-values for the current state.
        # No gradient is needed because this is action selection, not training.
        with torch.no_grad():
            qvals = agent.get_qvals(obs_stacked.to(device).float())
            

        # Epsilon-greedy action selection:
        # - With probability epsilon, choose a random action.
        # - Otherwise, choose the action with the highest predicted Q-value.
        eps = max(epsilon_end, epsilon_start - step * (epsilon_start - epsilon_end) / decay_steps)
        random_val = random.random()
        if eps > random_val:
            action = random.randrange(num_actions) #random action
        else:
            action = qvals.argmax(dim=-1).item() #neural network action selection (highest Q)

        action_selection = selection_to_action(action)

        # Accumulate reward across the skipped frames.
        reward = 0
        for i in range(frame_skip):
            next_obs, _, terminated, truncated, info = env.step(action_selection)

            #~~~~~~~~~~~~~calculate reward~~~~~~~~~~~~~#
            # The default environment reward is ignored here.
            # Instead, we build a reward from RAM values so Mario is rewarded for progress.
            ram = env.unwrapped.get_ram()
            won = ram[0x000D6]
            died = ram[0x000F1]
            far = int(ram[0x00023])
            x_hi = int(ram[0x0075])
            x_lo = int(ram[0x0090])

            r = 0

            # Small time penalty so the agent prefers finishing efficiently.
            r-=0.0025

            end_reward = 0
            # Reward level completion and penalize dying.
            if won == 216:
                terminated = True
                end_reward += 5.0
            elif died > 0:
                terminated = True
                end_reward -= 2.0

            # Reward Mario for reaching a new furthest x-position.
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

            # End the episode if Mario has not made progress for too long.
            if stagnant > 180:
                truncated = True

            reward += r
            reward += end_reward
            #~~~~~~~~~~~~~calculate reward~~~~~~~~~~~~~#

            done = terminated | truncated

            # Preprocess the next frame and update the frame stack.
            next_obs = cv2.cvtColor(cv2.resize(next_obs, (84, 84)), cv2.COLOR_BGR2GRAY)
            next_obs = torch.tensor(next_obs, dtype=torch.uint8).to("cpu")
            frame_stack.append(next_obs)
            next_obs_stacked = stack_obs(frame_stack)

            if done:
                break


        # Store the transition from this step.
        # This is the data the DQN will later sample from to learn.
        buffer.add([obs_stacked, action, reward, next_obs_stacked, done])

        episodic_return+=reward
        episode_length+=1

        if done:
            # Reset the environment at the end of an episode.
            next_obs, _ = env.reset()

            # Reset progress tracking for the next episode.
            ram = env.unwrapped.get_ram()
            x_hi = int(ram[0x0075])
            x_lo = int(ram[0x0090])
            maxX = x_hi*256 + x_lo
            stagnant = 0

            # Preprocess the first frame of the new episode.
            next_obs = cv2.cvtColor(cv2.resize(next_obs, (84, 84)), cv2.COLOR_BGR2GRAY)
            next_obs = torch.tensor(next_obs, dtype=torch.uint8).to("cpu")

            # Refill the frame stack with the first frame of the new episode.
            frame_stack.clear()
            for _ in range(k):
                frame_stack.append(next_obs)
            next_obs_stacked = stack_obs(frame_stack)


            # Log episode return so training progress can be viewed in TensorBoard.
            writer.add_scalar("charts/episodic_return", episodic_return, step)
            print(f"[Episode End] Step {step} | Return={episodic_return:.2f} | Len={episode_length}")

            episodic_return = 0
            episode_length = 0

        # Move to the next state.
        obs_stacked = next_obs_stacked



        # Only start learning once the replay buffer has enough data.
        # The step > 10000 condition gives the agent a warmup period of random-ish experience.
        if len(buffer.buffer) >= batch_size and step > 10000:

            # Sample a random batch of past transitions from the replay buffer.
            B = buffer.sample(batch_size)
            states, actions, rewards, new_states, dones = zip(*B)

            states = torch.stack(states).to(device).float() # (batch, 84, 84, 4)
            actions = torch.as_tensor(actions, dtype=torch.long, device=device) 
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            new_states = torch.stack(new_states).to(device).float() # (batch, 84, 84, 4)
            dones = torch.as_tensor(dones, dtype=torch.float32, device=device) 

            # Predict Q-values for every action in each sampled state.
            q_all = agent(states)

            # Select only the Q-value corresponding to the action that was actually taken.
            batch_indices = torch.arange(batch_size, device=device) #tensor([0,1,2,....,batch_size])
            q_sa = q_all[batch_indices, actions] #row 0 gets actions[0], row 1 gets actions[1]


            # Compute the TD target:
            #
            #   target = reward + gamma^frame_skip * max_a Q_target(next_state, a)
            #
            # If the episode ended, the future value term is removed.
            with torch.no_grad():
                next_q = target(new_states).max(dim=1).values
                td_target = rewards + (gamma ** frame_skip) * (1.0 - dones) * next_q


            # Huber loss is less sensitive to large errors than mean squared error.
            # This is commonly used in DQN-style training.
            loss = F.smooth_l1_loss(q_sa, td_target)

            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to reduce the chance of unstable updates.
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 10.0)

            optimizer.step()

            # Hard update: periodically copy the online network weights to the target network.
            if step % target_update_freq == 0:
                target.load_state_dict(agent.state_dict())


    env.close()
