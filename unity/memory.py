import math
import numpy as np
from utils import get_gaes
from collections import defaultdict



class Trajectory():
    def __init__(self):
        self.trajectory_metric = ["obs", "actions", "log_probs", "rewards", "v_preds", "dones"]
        # Current going trajectory
        # metric -> id -> np
        self.cur_trajectory = {each_str : defaultdict(list) for each_str in self.trajectory_metric}
        self.history_length_each_agent = defaultdict(lambda:0)    

        # Add one step for all agents
    def add_one_step(self, obs, action, log_prob, reward, v_preds):
        for agent_id in obs.keys():
            self.add_one_step_for_one_agent(agent_id, obs, action, log_prob, reward, v_preds)

    # Add one step for one agent. Designed for done agent.
    def add_one_step_for_one_agent(self, agent_id, obs, action, log_prob, reward, v_preds, done=False):
        self.cur_trajectory["obs"][agent_id].append(obs[agent_id])
        self.cur_trajectory["actions"][agent_id].append(action[agent_id])
        self.cur_trajectory["log_probs"][agent_id].append(log_prob[agent_id])
        self.cur_trajectory["rewards"][agent_id].append(reward[agent_id])
        self.cur_trajectory["v_preds"][agent_id].append(v_preds[agent_id].detach())
        self.cur_trajectory["dones"][agent_id].append(done)
        self.history_length_each_agent[agent_id] += 1
    
    def clear_agent(self, agent_id):
        for item in self.trajectory_metric:
            self.cur_trajectory[item][agent_id].clear()
            self.history_length_each_agent[agent_id] = 0

    def __getitem__(self, index):
        return self.cur_trajectory[index]
    
    def keys(self):
        return self.trajectory_metric
    


class Memory():
    def __init__(self, logger, gamma, lam):
        self.batch_metric = ["obs", "actions", "log_probs", "rewards", "v_preds", "next_v_preds", "gaes", "dones"] # Remove "done"
        self.learn_metric = ["obs", "actions", "log_probs", "next_v_preds", "rewards", "gaes"]

        # Total trajectory memory, will be updated 
        self.batch_data = {each_str : defaultdict(list) for each_str in self.batch_metric}
        self.history_length_each_agent = defaultdict(lambda:0)    

        # tensorboard logger
        self.logger = logger

        self.gamma = gamma
        self.lam = lam

        self.episode_num = 0

    # Add one trajectory to memory
    def add_to_batch(self, trajectory, agent_id):
        # Obtain next_v_preds and gaes from running trajectory
        next_v_preds =trajectory["v_preds"][agent_id] + [0]
        gaes =  get_gaes(trajectory["rewards"][agent_id], trajectory["v_preds"][agent_id], next_v_preds, self.gamma, self.lam)
        gaes = np.array(gaes).astype(dtype=np.float64)
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-5 ) 

        self.batch_data["next_v_preds"][agent_id] += next_v_preds
        self.batch_data["gaes"][agent_id] += list(gaes)

        self.history_length_each_agent[agent_id] += trajectory.history_length_each_agent[agent_id]

        for item in trajectory.keys():
            self.batch_data[item][agent_id] += trajectory[item][agent_id]

        self.episode_num += 1
        info = {
            "episode_max_reward" : max(trajectory["rewards"][agent_id]),
            "mean_reward" : sum(trajectory["rewards"][agent_id]) / len(trajectory["rewards"][agent_id]),
            "episode_length" : len(trajectory["rewards"][agent_id]),
            "episode_reward" : sum(trajectory["rewards"][agent_id]),
            "max_pi" : math.exp(max(trajectory["log_probs"][agent_id])),
        }

        for key, value in info.items():
            self.logger.write_tb(key, value, self.episode_num)

        trajectory.clear_agent(agent_id)

    # Merge all agents' data to get one batch.
    # NOTE:: This function just suits all agents share the same parameters.
    def get_one_batch(self, batch_size):
        data = []
        if sum(self.history_length_each_agent.values()) >= batch_size:
            for item in self.learn_metric:
                item_data = self.batch_data[item]
                data.append(np.concatenate([ item for item in item_data.values() if len(item) > 0]))
            # Clean current memory.
            self.refresh()
        else:
            return None
        return data
    
    def refresh(self):
        for agent_id in self.history_length_each_agent.keys():
            self.history_length_each_agent[agent_id] = 0
        
        for each_str in self.batch_metric:
            for agent_id in self.batch_data[each_str].keys():
                self.batch_data[each_str][agent_id] = []

