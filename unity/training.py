import argparse
import numpy as np
from env.gym_wrapper import unityEnv
from ppo import PPO
from collections import defaultdict


def main(args):
    # Create environment
    env = unityEnv(args.binPath)
    # Get Observation
    cur_obs = env.reset()

    action_shape = env.action_shape
    obs_shape = env.observation_shape
    alg = PPO(obs_shape, action_shape)
    
    memory = { each_str : defaultdict(list) for each_str in ["obs", "actions", "log_probs", "rewards", "v_preds", "next_v_preds", "dones"]}

    for step in range(int(1e3)):
        action = {}

        action, value, log_prob = alg.act(cur_obs)
        next_obs, reward, done, info = env.step(action)

        if not done.keys():
            if len(cur_obs) == env.n_agents:
                for agent_id in env.n_agent_ids:
                    memory["obs"][agent_id].append(cur_obs[agent_id])
                    memory["rewards"][agent_id].append(reward[agent_id])
                    memory["v_preds"][agent_id].append(value[agent_id].detach())
                    memory["actions"][agent_id].append(action[agent_id])
                    memory["log_probs"][agent_id].append(log_prob[agent_id])
                    memory["dones"][agent_id].append(False)
        else:
            data = []
            for agent_id in done.keys():
                memory["next_v_preds"][agent_id] = memory["v_preds"][agent_id][1:] + [0]
                gaes = alg.get_gaes(memory["rewards"][agent_id], memory["v_preds"][agent_id], memory["next_v_preds"][agent_id])
                gaes = np.array(gaes).astype(dtype=np.float64)
                gaes = (gaes - gaes.mean()) / gaes.std()
                # data = [obs, actions, log_probs, next_v_preds, rewards, gaes]
                data = [memory[each][agent_id] for each in ["obs", "actions", "log_probs", "next_v_preds", "rewards"]] + [gaes]

                for _ in range(alg.n_update):
                    # Sample training data
                    sample_indices = np.random.randint(low=0, high=len(memory["rewards"][agent_id]), size=alg.batch_size)
                    sampled_data = [np.take(a=a, indices=sample_indices, axis=0) for a in data]

                    # # Train model
                    alg.learn(*sampled_data)

            print("Learned!")

        cur_obs = next_obs

        print(f"step = {step} obs = {len(cur_obs)} reward = {reward} done = {done.keys()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--binPath', type=str, default='./env/aircraftLearning.app')
    args = parser.parse_args()
    main(args)




