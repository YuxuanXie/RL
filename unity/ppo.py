import torch
from torch import nn
import torch.optim as optim
from torch import distributions
from torch.autograd import Variable
from model import Model
from utils import get_dist



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
            config=None,
            gpu_num=1):

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

        model_config = config["model"] if "model" in config.keys() else None
        # Model
        self.model = Model(observation_shape, action_shape, model_config)
        # self.model.apply(self.weights_init)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        

        # Cuda 
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_num}')
            torch.cuda.set_device(gpu_num)
            self.model.cuda()
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.device = torch.device('cpu')
            self.FloatTensor = torch.FloatTensor


    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)


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
            # action[agent_id] = [d.sample() for d in dist]
            action[agent_id] = [ p.argmax() for p in probs]
            # log_prob[agent_id] = sum([d.log_prob(a) for d, a in zip(dist, action[agent_id])]).item()
            log_prob[agent_id] = 0.0

        return action, values, log_prob

    # Used for updating the model
    # Input: 
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
        obs = Variable(self.FloatTensor(obs))
        actions = Variable(self.FloatTensor(actions))
        rewards = Variable(self.FloatTensor(rewards))
        gaes = Variable(self.FloatTensor(gaes))
        log_probs = Variable(self.FloatTensor(log_probs))
        next_v_preds = Variable(self.FloatTensor(next_v_preds))

        new_log_probs, state_values, entropy = self.evaluate_actions(obs, actions)
        ratio = torch.exp(new_log_probs - log_probs)
        clipped_ratios = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)

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

    def load_model(self, path, cuda=True):
        # self.model.load_state_dict(torch.load(path))
        if cuda:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_steps = checkpoint['update_steps']


