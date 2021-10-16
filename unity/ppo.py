import copy
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch import distributions
import torch.nn.functional as F
from torch.autograd import Variable


class model(nn.Module):
    def __init__(self, observation_shape, action_shape, hidden_size=256):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.hidden_layers = [hidden_size, int(hidden_size/2)]
        
        self.f1 = nn.Linear(self.observation_shape, self.hidden_layers[0])
        self.f2 = nn.Linear(self.hidden_layers[0], self.hidden_layers[1])
        # Action head
        self.logits = [nn.Linear(self.hidden_layers[1], output_size) for output_size in self.action_shape]

        # Value head
        self.value = nn.Sequential(
            nn.Linear(self.observation_shape, self.hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[1], 1)
        )


    def forward(self, obs):
        x = F.relu(self.f1(obs))
        x = F.relu(self.f2(x))
        logits = [head(x) for head in self.logits]
        probs = [F.softmax(each, dim=-1) for each in logits]
        value = self.value(obs)
        return probs, value


class PPO():
    def __init__(self, 
            observation_shape,
            action_shape,
            learning_rate=1e-5,
            gamma = 0.9,
            lam = 0.8,
            batch_size = 128,
            n_updates = 4,
            clip_ratio = 0.3,
            c1 = 1.0,
            c2 = 0.00,
            hidden_size=128):

        # Env parameters
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        
        # Hyper parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.n_update = n_updates
        self.clip_ratio = clip_ratio
        self.c1 = c1
        self.c2 = c2
        self.update_steps = 0
        self.gradient_nrom = 10

        # Model
        self.model = model(observation_shape, action_shape, hidden_size)
        # self.model.apply(self.weights_init)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)

    # Return distributions of all action dimension 
    # For all agents
    # Input : list of probs
    # Return : list of distributions
    def get_dist(self, probs):
        dist = [distributions.Categorical(each) for each in probs ]
        return dist
    
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
            dist = self.get_dist(probs)
            values[agent_id] = v[0]
            action[agent_id] = [d.sample() for d in dist]
            log_prob[agent_id] = sum([d.log_prob(a) for d, a in zip(dist, action[agent_id])]).item()

        return action, values, log_prob

    # For one agent
    def get_gaes(self, rewards, v_preds, next_v_preds):
        # source: https://github.com/uidilr/ppo_tf/blob/master/ppo.py#L98
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, next_v_preds, v_preds)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.lam * self.gamma * gaes[t + 1]
        return gaes

    # For one agent
    # Input obs : n * self.observation_shape
    # Input actions : n * self.action_shape

    # def evaluate_actions(self, obs, actions):
    #     outputs, values = self.model(obs)
    #     # Get distributions for tree dimension  dist[n]-> action_dimension n 
    #     dist = np.array([self.get_dist(output) for output in outputs])
    #     dist = dist.transpose()
    #     log_prob = []
    #     entropy = []
    #     for d, a in zip(dist, actions):
    #         log_prob.append(np.prod([each_d.log_prob(each_a) for each_d, each_a in zip(d,a)]))
    #         entropy.append(sum([each_d.entropy() for each_d in d])/len(d))

    #     return torch.stack(log_prob), values, torch.stack(entropy)

    def evaluate_actions(self, obs, actions):
        outputs, values = self.model(obs)
        # Get distributions for tree dimension  dist[n]-> action_dimension n 
        dist = [distributions.Categorical(output) for output in outputs]
        actions = actions.transpose(0,1)
        log_prob = []
        entropy = []
        for d, a in zip(dist, actions):
            log_prob.append(d.log_prob(a))
            entropy.append(d.entropy())

        entropy = sum(entropy) / len(entropy)
        log_prob = sum(log_prob)

        return log_prob, values, entropy

    # For one agent
    def learn(self, obs, actions, log_probs, next_v_preds, rewards, gaes):
        obs = Variable(torch.FloatTensor(obs))
        actions = Variable(torch.FloatTensor(actions))
        rewards = Variable(torch.FloatTensor(rewards))
        gaes = Variable(torch.FloatTensor(gaes))
        log_probs = Variable(torch.FloatTensor(log_probs))

        new_log_probs, state_values, entropy = self.evaluate_actions(obs, actions)
        ratio = torch.exp(new_log_probs - log_probs)
        clipped_ratios = torch.clip(ratio, 1-self.clip_ratio, 1+self.clip_ratio)

        loss_clip = torch.min(gaes*ratio, gaes*clipped_ratios)
        loss_clip = torch.mean(loss_clip)

        target_values = rewards + self.gamma * next_v_preds
        vf_loss = torch.mean(torch.square(state_values - target_values))
        entropy = torch.mean(entropy)

        total_loss = - loss_clip + self.c1 * vf_loss - self.c2 * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_nrom)
        self.optimizer.step()

        self.update_steps += 1

        return total_loss, loss_clip, vf_loss, entropy

    def save_model(self, path):
        # torch.save(self.model.state_dict(), path)
        torch.save({
            'update_steps': self.update_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        # self.model.load_state_dict(torch.load(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_steps = checkpoint['update_steps']


