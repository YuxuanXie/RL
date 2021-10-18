import os
import math
import argparse
from datetime import datetime
import numpy as np
from env.gym_wrapper import unityEnv
from ppo import PPO
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

time_token = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
tblog_dir = f"./results/tblog/{time_token}/"
model_dir = f"./results/model/{time_token}/"
writer = SummaryWriter(log_dir=tblog_dir)
os.makedirs(model_dir)


# TODO: Add rollout and datamanager to make the structure clearer.

def main(args):
    # Create environment
    env = unityEnv(args.binPath)

    # Get Observation
    cur_obs = env.reset()
    action_shape = env.action_shape
    obs_shape = env.observation_shape
    alg = PPO(obs_shape, action_shape, n_updates=4, learning_rate=5e-5, batch_size=256, clip_ratio=0.3, hidden_size=128, c1=1.0, c2=0.01, gamma=0.99, lam=1.0)
    
    memory = { each_str : defaultdict(list) for each_str in ["obs", "actions", "log_probs", "rewards", "v_preds", "next_v_preds", "dones"]}
    data = [[] for _ in range(6)]

    for episode in range(int(1e7)):
        action = {}

        action, value, log_prob = alg.act(cur_obs)
        next_obs, reward, done, info = env.step(action)

        if not done.keys():
            if len(cur_obs) == env.n_agents:
                for agent_id in env.n_agent_ids:
                    memory["obs"][agent_id].append(cur_obs[agent_id])
                    memory["actions"][agent_id].append(action[agent_id])
                    memory["log_probs"][agent_id].append(log_prob[agent_id])
                    memory["rewards"][agent_id].append(reward[agent_id])
                    memory["v_preds"][agent_id].append(value[agent_id].detach())
                    memory["dones"][agent_id].append(False)

        else:
            # if len(data[0]) > alg.batch_size:
            data = [[] for _ in range(6)]
            for agent_id in done.keys():
                
                if agent_id in cur_obs.keys(): 
                    memory["obs"][agent_id].append(cur_obs[agent_id])
                    memory["actions"][agent_id].append(action[agent_id])
                    memory["log_probs"][agent_id].append(log_prob[agent_id])
                    memory["rewards"][agent_id].append(reward[agent_id])
                    memory["v_preds"][agent_id].append(value[agent_id].detach())
                    memory["dones"][agent_id].append(True)

                memory["next_v_preds"][agent_id] = memory["v_preds"][agent_id][1:] + [0]
                gaes = alg.get_gaes(memory["rewards"][agent_id], memory["v_preds"][agent_id], memory["next_v_preds"][agent_id])
                gaes = np.array(gaes).astype(dtype=np.float64)
                gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-5 ) 
                
                # Concate two list
                temp = [memory[each][agent_id] for each in ["obs", "actions", "log_probs", "next_v_preds", "rewards"]] + [gaes]
                data = [list(d) + list(t) for d, t in zip(data, temp)]

                if len(memory["rewards"][agent_id]) > 0 :
                    writer.add_scalar("Info/episode_max_reward", max(memory["rewards"][agent_id]), episode)
                    writer.add_scalar("Info/mean_reward", sum(memory["rewards"][agent_id])/len(memory["rewards"][agent_id]), episode)
                    writer.add_scalar("Info/episode_length", len(memory["rewards"][agent_id]), episode)
                    writer.add_scalar("Info/episode_reward", sum(memory["rewards"][agent_id]), episode)
                    writer.add_scalar("Info/max_pi", math.exp(max(memory["log_probs"][agent_id])), episode)

                if sum(memory["rewards"][agent_id]) > 4.0 :
                    env.set_env_parameters(value=max(0, env.cirrculum_param - 5))

                for key in memory.keys():
                    memory[key][agent_id].clear()

            for i in range(int(len(data[0]) * alg.n_update / alg.batch_size)):
                # Sample training data
                sample_indices = np.random.randint(low=0, high=len(data[0]), size=alg.batch_size)
                sampled_data = [np.take(a=a, indices=sample_indices, axis=0) for a in data]

                # # Train model
                loss, policy_loss, value_loss, entropy = alg.learn(*sampled_data)
                if i == 0:
                    writer.add_scalar("Info/total_loss", loss.item(), alg.update_steps)
                    writer.add_scalar("Info/policy_loss", policy_loss.item(), alg.update_steps)
                    writer.add_scalar("Info/value_loss", value_loss.item(), alg.update_steps)
                    writer.add_scalar("Info/entropy", entropy.item(), alg.update_steps)

        if episode % 100000 == 0:
            alg.save_model(model_dir + f'{episode}.pth')


        # print(f"step = {step} obs = {len(cur_obs)} reward = {reward} done = {done.keys()} update = {int(len(data[0]) * alg.n_update / alg.batch_size)} ")
        cur_obs = next_obs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--binPath', type=str, default='./env/aircraftLearning.app')
    args = parser.parse_args()
    main(args)




