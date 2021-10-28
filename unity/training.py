import math
import logging
import argparse
import numpy as np
from numpy.core.fromnumeric import mean
from env.gym_wrapper import unityEnv
from ppo import PPO
from rollout import Rollout
from memory import Memory
from utils import TbLogger
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO)
# TODO: Add rollout and datamanager to make the structure clearer.

def main(args):
    # Create environment
    def env_creator():
        return unityEnv(args.binPath)

    model_config = {"hidden_size" : 128}
    # tb logger    
    logger = TbLogger()
    # Central data manager
    memory = Memory(logger, args.gamma, args.lam)
    # Rollout worker
    worker = Rollout(env_creator, memory, model_config)
    # PPO learner
    alg = PPO(worker.obs_shape, worker.action_shape, n_updates=args.n_updates, learning_rate=args.learning_rate, batch_size=args.batch_size, clip_ratio=args.clip_ratio, c1=args.c1, c2=args.c2, gamma=args.gamma, lam=args.lam, model_config=model_config)

    while alg.update_steps < 1e7:
        worker.run()
        data = memory.get_one_batch(alg.batch_size)
        if data:
            for i in range(int(len(data[0]) * alg.n_update / alg.batch_size)):
                # Sample training data
                sample_indices = np.random.randint(low=0, high=len(data[0]), size=alg.batch_size)
                sampled_data = [np.take(a=a, indices=sample_indices, axis=0) for a in data]
                # # Train model
                loss, policy_loss, value_loss, entropy = alg.learn(*sampled_data)
                if i == 0:
                    logger.write_tb("total_loss", loss.item(), alg.update_steps)
                    logger.write_tb("policy_loss", policy_loss.item(), alg.update_steps)
                    logger.write_tb("value_loss", value_loss.item(), alg.update_steps)
                    logger.write_tb("entropy", entropy.item(), alg.update_steps)

            logging.debug(f"learned {len(data[0])} steps!")
            worker.update_model(alg.model)

        if logger.save_model(alg.update_steps, 1e5):
            alg.save_model(logger.model_dir + f'{alg.update_steps}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--binPath', type=str, default='./env/al_linux.x86_64')
    parser.add_argument('--n_updates', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--c1', type=float, default=1.0)
    parser.add_argument('--c2', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lam', type=float, default=0.9)
    args = parser.parse_args()
    main(args)




