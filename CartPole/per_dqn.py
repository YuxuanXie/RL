import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable

from datetime import datetime
from per.prioritized_memory import Memory


class Net(nn.Module):
    def __init__(self, state_space, action_space, hidden_layer=64) -> None:
        super().__init__()
        self.hidden_layer = hidden_layer
        self.l1 = nn.Linear(state_space, self.hidden_layer)
        self.l2 = nn.Linear(self.hidden_layer, action_space)

    def forward(self, s):
        x = F.relu(self.l1(s))
        return self.l2(x)

class DQN():
    def __init__(self, state_space, action_space) -> None:
        self.state_space = state_space
        self.action_space = action_space

        self.render = False
        self.logdir = f"./log/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        self.load_model_dir = ''
        # Hyper parameters
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.explore_step = 5000
        self.memory_size = int(1e5)
        self.learning_rate = 1e-4
        self.epsilon_decay = (self.epsilon - self.epsilon_min)/self.explore_step
        self.batch_size = 64
        self.update_target_steps = 200

        self.memory = Memory(self.memory_size)

        # Create model
        self.model = Net(state_space, action_space)
        self.model.apply(self.weights_init)
        self.target_model = Net(state_space, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.learning_steps = 0

        # Initialize the target model
        self.update_target_model()

        if self.load_model_dir:
            self.model = torch.load(self.load_model_dir)


    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Output action
    def get_action(self, s):
        if random.uniform(0,1) < self.epsilon:
            action = env.action_space.sample()
        else:
            state = torch.from_numpy(s)
            state = Variable(state).float().cpu()
            qs = self.model(state)
            _, action = torch.max(qs, -1)
            action = action.item()
        return action

    # Save the (s, a, r, s', done) to the replay buffer
    def add_sample(self, state, action, reward, state_next, done):
        q = self.target_model(Variable(torch.FloatTensor(state)))
        q_next = self.target_model(Variable(torch.FloatTensor(state_next)))

        target = reward + (1-int(done)) * self.discount_factor * torch.max(q_next)

        error = torch.abs(q[0][action] - target)

        self.memory.add(error.detach(), (state, action, reward, state_next, done))
    
    def train(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        self.learning_steps += 1
        if self.learning_steps % self.update_target_steps == 0:
            self.update_target_model()

        batch, idx, is_weight = self.memory.sample(self.batch_size)
        batch = np.array(batch).transpose()

        states = np.vstack(batch[0])
        actions = list(batch[1])
        rewards = list(batch[2])
        state_nexts = np.stack(batch[3])
        dones = batch[4]

        dones = dones.astype(int)
        
        states = Variable(torch.FloatTensor(states)).float()
        onehot_actions = torch.FloatTensor(self.batch_size, self.action_space).zero_()
        onehot_actions.scatter_(1, torch.LongTensor(actions).view(-1, 1), 1)

        q = self.model(Variable(torch.FloatTensor(states)))
        q = torch.sum(q.mul(Variable(onehot_actions)), dim=-1)

        q_next = self.target_model(Variable(torch.FloatTensor(state_nexts))).detach()
        q_next_max, _ = torch.max(q_next, -1)
        
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        targets = rewards + (1-dones) * self.discount_factor * q_next_max.squeeze()
        targets = Variable(targets)

        error = torch.abs(q - targets).data.numpy()

        for i, priority_id in enumerate(idx):
            self.memory.update(priority_id, error[i])
        
        loss = (torch.FloatTensor(is_weight) * F.mse_loss(q, targets)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQN(state_size, action_size)

    for e in range(int(1e3)):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            if agent.render:
                env.render()
            
            action = agent.get_action(state)
            # import pdb; pdb.set_trace()
            state_next, reward, done, info = env.step(action)
            state_next = np.reshape(state_next, [1, state_size])
            agent.add_sample(state, action, reward, state_next, done)
            state = state_next

            score += reward

            if agent.memory.tree.n_entries >= agent.batch_size:
                agent.train()

            if done:
                print(f"episode = {e} score = {score}")






