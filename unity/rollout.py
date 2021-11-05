import torch
import logging
from model import Model
from memory import Trajectory
from torch.autograd import Variable
from utils import get_dist, CLManager


class Rollout():
    def __init__(self, env_creator, memory, config):
        self.env = env_creator()
        self.cur_obs = self.env.reset()
        self.action_shape = self.env.action_shape
        self.obs_shape = self.env.observation_shape
        self.model = Model(self.obs_shape, self.action_shape, config["model"])
        self.trajectory = Trajectory()
        self.data_manager = memory
        self.CL_manager = CLManager(config["CL"]["threshold"], config["CL"]["CL_params"])
        self.env.set_env_parameters(value = self.CL_manager.get_cur_CL_params())
    
    # Generate trajectories
    # TODO: fix the issus about droping one step when one agent is done.
    def run(self):
        for episode in range(int(1e7)):

            action, v_preds, log_prob = self.act(self.cur_obs)
            next_obs, reward, done, info = self.env.step(action)

            if not done.keys():
                if len(self.cur_obs) == self.env.n_agents:
                    self.trajectory.add_one_step(self.cur_obs, action, log_prob, reward, v_preds)
            else:
                for agent_id in done.keys():
                    if agent_id in self.cur_obs.keys(): 
                        self.trajectory.add_one_step_for_one_agent(agent_id, self.cur_obs, action, log_prob, reward, v_preds, done=True)
                        self.CL_manager.add_reward(sum(self.trajectory["rewards"][agent_id]))
                        # Add agent_id's trajectory data into the memory and clear
                        # NOTE: All trajectory operations have to be done before this !!
                        self.data_manager.add_to_batch(self.trajectory, agent_id)

                if self.CL_manager.to_next_level():
                    self.env.set_env_parameters(value = self.CL_manager.get_cur_CL_params())

            self.cur_obs = next_obs
            logging.debug(f"step = {episode} obs = {len(self.cur_obs)} reward = {reward} done = {done.keys()}")

            if done.keys(): return

    # Get action for each obs
    # For all agents
    # Input : dict id->obs
    # Return : dict id -> obj
    def act(self, obs):
        action = {}
        log_prob = {}
        values = {}
        for agent_id in obs.keys():
            probs, v = self.model(Variable(torch.FloatTensor(obs[agent_id])))
            dist = get_dist(probs)
            values[agent_id] = v[0]
            action[agent_id] = [d.sample() for d in dist]
            log_prob[agent_id] = sum([d.log_prob(a) for d, a in zip(dist, action[agent_id])]).item()

        return action, values, log_prob
    
    def update_model(self, new_model):
        self.model.load_state_dict(new_model.state_dict())

