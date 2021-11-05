import os
import math
import argparse
from datetime import datetime
import numpy as np
from env.gym_wrapper import unityEnv
from ppo import PPO
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


def main(args):


    config = {
        "model" :  {"hidden_size" : 128},
        "CL" : {
            "threshold" : [2, 2, 4, 6],
            "CL_params" : [50, 30, 15, 0]
        },
    }

    # Create environment
    env = unityEnv(args.binPath)
    # Get Observation
    cur_obs = env.reset()
    env.set_time_scale(time_scale=1)

    action_shape = env.action_shape
    obs_shape = env.observation_shape
    alg = PPO(obs_shape, action_shape, config=config)
    
    alg.load_model(args.checkpoint, cuda=False)

    for step in range(int(1e3)):
        action = {}

        action, value, log_prob = alg.act(cur_obs)
        next_obs, reward, done, info = env.step(action)

        print(f"step = {step} obs = {len(cur_obs)} reward = {reward} done = {done.keys()}")
        cur_obs = next_obs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--binPath', type=str, default='./env/visuallizer.app')
    parser.add_argument('--checkpoint', type=str, default='results/model/2021-10-17-11-59-50/900000.pth')
    args = parser.parse_args()
    main(args)




