import numpy as np
from .multiagentenv import MultiAgentEnv
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

class unityEnv(MultiAgentEnv):
    "The gym wrapper for multi-agent unity environemnt"
    def __init__(self, 
                bin_path,
                base_port=5005,
                worker_id=0,
                ):
        """
        Create a unity environment 

        ------
        Parameters:
            bin_path 
        
        """
        # Env args
        self.name = bin_path.replace(".app", "").split('/')[-1]
        self.engine_config_channel = EngineConfigurationChannel()
        self.parameter_config_channel = EnvironmentParametersChannel()

        self._env = UnityEnvironment(file_name=bin_path, side_channels=[self.engine_config_channel, self.parameter_config_channel], base_port=base_port, worker_id=worker_id )
        self._env.reset()

        # Observation args
        self.behavior_name = list(self._env.behavior_specs._dict.keys())[0]
        self.spec = self._env.behavior_specs[self.behavior_name]
        self.observation_shape = sum([ each[0] for each in self.spec.observation_shapes])

        # Action args
        self.discrete = self.spec.is_action_discrete()
        self.action_shape = self.spec.action_shape

        # Agent args
        self.decision_steps, self.terminal_steps = self._env.get_steps(self.behavior_name)
        self.n_agents = len(self.decision_steps)
        self.n_agent_ids = self.decision_steps.agent_id
        
        # Use side channel to config unity env
        self.cirrculum_param = 50
        self.set_time_scale()
        self.set_env_parameters(value=self.cirrculum_param)
        


    def _extract_decision_info(self, decision_steps, terminated=False):
        # id -> numpy arry and its length is self.observation_shape
        obs = {}
        # id -> float
        reward = {}
        # id -> arrary whos dimension is equal to the action
        action_mask = {}
        for ida in decision_steps.agent_id:
            agent_info = decision_steps[ida]
            obs[ida] = np.concatenate([ e for e in agent_info.obs], axis=0)
            reward[ida] = agent_info.reward
            if not terminated:
                action_mask = agent_info.action_mask

        return obs, reward, action_mask


    def step(self, actions):
        """ Returns reward, terminated, info : type dict {id -> object} """

        # Set action for agents who asked
        
        for ida in actions.keys():
            self._env.set_action_for_agent(self.behavior_name, ida, np.array(actions[ida]))

        # Execute action
        self._env.step()

        # Get next state, reward, done, info
        info = {}
        self.decision_steps, self.terminal_steps = self._env.get_steps(self.behavior_name)
        obs, reward, _ = self._extract_decision_info(self.decision_steps)
        done_agents = self._extract_decision_info(self.terminal_steps, terminated=True)
        done = {agent_id : True for agent_id in self.terminal_steps.agent_id}
        # obs.update(done_agents[0])
        reward.update(done_agents[1])
        # import pdb; pdb.set_trace();
        
        # Get obs, reward, done, info for each agent
        return obs, reward, done, info


    def get_obs(self):
        """ Returns agent observations in a dict {id -> obs} """
        obs, _, _ = self._extract_decision_info(self.decision_steps)
        return obs
        
    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise self.observation_shape
    
    def get_action_size(self):
        """ Returns the shape of the action """
        return self.action_shape

    def reset(self):
        """ Returns initial observations"""
        self._env.reset()
        self.decision_steps, _ = self._env.get_steps(self.behavior_name)
        obs, _, _ = self._extract_decision_info(self.decision_steps)
        return obs

    def close(self):
        self._env.close()

    def get_env_info(self):
        env_info = {
                    "n_agents": self.n_agents,
                    }
        return env_info
    
    # Literally, the number of executed action doesnot change along with the time scale .... 
    # But the physics object solver might be inaccurate when the time scale is too large.
    def set_time_scale(self, time_scale=20):
        self.engine_config_channel.set_configuration_parameters(time_scale=time_scale)
    
    # Support float only
    def set_env_parameters(self, key="checkpoint_radius", value=50):
        print(f"Set {key} to {value}")
        self.parameter_config_channel.set_float_parameter(key, value)


def main():
    env = unityEnv()


if __name__ == '__main__':
    main()
