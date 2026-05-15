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
TUTORIAL PART 1 README
~~~~~~~~~~~~~~~~~~~~~~~~~

This is just a basic loop where mario plays the game completely randomly. 

It has the complete reward function included as well.

No neural network is added yet here.

Mario selects random actions until he dies or the game times out.

'''


#Uses GPU if available (cuda)
#Otherwise uses CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)



if __name__ == "__main__":

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #retro environment creation
    base_env = retro.make(game='SuperMarioBros3-Nes', render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(base_env)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="random_play_videos",
        #recording EVERY episode. Don't let this run too long.
        episode_trigger=lambda t: t % 1 == 0,
        fps=60
    )


    obs, _ = env.reset()

    num_steps = 10000000

    #hold an action for 4 frames
    frame_skip = 4

    #ram values for mario's x position (how we will calculate reward)
    ram = env.unwrapped.get_ram()
    x_hi = int(ram[0x0075])
    x_lo = int(ram[0x0090])
    #maxX is the highest X mario has reached per episode
    maxX = x_hi*256 + x_lo

    #stagnant measures how long mario has not progressed. If he does not progress in the time limit, we will reset early.
    stagnant = 0

    #end after 10 videos
    end_when_10 = 0

    for step in range(1, num_steps):


        #no neural network to select an action.. just random selection!
        action_selection = env.action_space.sample()

        reward = 0

        #hold action for each frame_skip step
        for i in range(frame_skip):


            #next_obs is the new state s'
            next_obs, _, terminated, truncated, info = env.step(action_selection)

            #~~~~~~~~~~~~~calculate reward~~~~~~~~~~~~~#
            ram = env.unwrapped.get_ram()
            #won: True if mario won
            won = ram[0x000D6]
            #died: True if mario died
            died = ram[0x000F1]
            #far: another related x position
            far = int(ram[0x00023])
            #x_hi and x_lo are combined for the true X_position
            x_hi = int(ram[0x0075])
            x_lo = int(ram[0x0090])

            # Small time penalty so the agent prefers finishing efficiently.
            r = 0
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
            if stagnant > 180:
                truncated = True

            reward += r
            reward += end_reward

            #~~~~~~~~~~~~~calculate reward~~~~~~~~~~~~~#

            #terminated means death, truncated means episode cut short due to timeout
            done = terminated | truncated

            #get out of frame_skip if game is over
            if done:
                break

        #if the episode has ended, reset the game and all helper values
        if done:
            next_obs, _ = env.reset()
            ram = env.unwrapped.get_ram()
            x_hi = int(ram[0x0075])
            x_lo = int(ram[0x0090])
            maxX = x_hi*256 + x_lo
            stagnant = 0
            end_when_10 += 1
        
        #end after 10 videos
        if end_when_10 == 10:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("10 random episodes complete! Exiting!")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            exit(1)


    env.close()