import os
import copy
from datetime import datetime
from torch import distributions
from torch.utils.tensorboard import SummaryWriter


class TbLogger():
    def __init__(self):
        self.time_token = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.tblog_dir = f"./results/tblog/{self.time_token}/"
        self.model_dir = f"./results/model/{self.time_token}/"
        self.writer = SummaryWriter(log_dir=self.tblog_dir)
        self.previous_save_model_step = 0

        os.makedirs(self.model_dir)
    
    def write_tb(self, tag, value, step):
        self.writer.add_scalar(f"Info/{tag}", value, step)
    
    def save_model(self, step, interval):
        if step - self.previous_save_model_step > interval:
            self.previous_save_model_step = step
            return True
        else:
            return False


# For one agent
def get_gaes(rewards, v_preds, next_v_preds, gamma, lam):
    # source: https://github.com/uidilr/ppo_tf/blob/master/ppo.py#L98
    deltas = [r_t + gamma * v_next - v for r_t, v_next, v in zip(rewards, next_v_preds, v_preds)]
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + lam * gamma * gaes[t + 1]
    return gaes

# Return distributions of all action dimension 
# For all agents
# Input : list of probs
# Return : list of distributions
def get_dist(probs):
    dist = [distributions.Categorical(each) for each in probs ]
    return dist