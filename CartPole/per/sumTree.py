import numpy as np

class sumTree():

    def __init__(self, capacity):
        self.capacity = capacity
        self.writer = 0
        self.tree = np.zeros(2*self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)
        self.n_entreis = 0


    # Return the total sum, which is the value of the root
    def total(self):
        return self.tree[0]

    # Change the priority from the leaf to the root
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    # Fetch sample on the leaf node
    def _retrive(self, idx, s):
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left >= 2*self.capacity - 1:
            return left
        
        if self.tree[left] >= s:
            self._retrive(left, s)
        else:
            self._retrive(right, s - self.tree[left])

    # Add new data to the sum tree based on the priority
    # p : priority
    # data : transition
    def add(self, p, data):
        idx = self.writer + self.capacity - 1

        self.data[self.writer] = data
        self.update(idx, p)

        self.writer += 1
        if self.writer >= self.capacity:
            self.writer = 0
        
        if self.n_entreis < self.capacity:
            self.n_entreis += 1

    # Update the priority
    def update(self, idx, p)
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)


    # Get data
    # s : sum
    def get(self, s):
        idx = self._retrive(0,s)
        return (idx, self.tree[idx], self.data[idx - self.capacity + 1])

        