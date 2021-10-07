import random
import numpy as np
from . import sumTree


class Memory():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 1e-3
        self.tree = sumTree.sumTree(capacity)

    def _get_priority(self, error):
        return (np.abs(error) + self.e)**self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    # Get n samples from the replay buffer
    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = min([1, self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i+1)

            s = random.uniform(a,b)
            (idx, p, data) = self.tree.get(s)
            if p == 0.0:
                print("replace")
                idx, p, data = self.tree.n_entries-1+self.capacity-1, self.tree.tree[self.tree.n_entries-1+self.capacity-1], self.tree.data[self.tree.n_entries-1]
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        
        sample_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sample_probabilities, -self.beta)
        is_weight /= is_weight.max()


        return batch, idxs, is_weight
    
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
    


    