
from collections import defaultdict

class Memory():
    def __init__(self, capacity):
        self.metric = ["obs", "actions", "log_probs", "rewards", "v_preds", "next_v_preds", "dones", "gae"]
        self.batch_data = {each_str : defaultdict(list) for each_str in self.metric}

    def add_one_step()
