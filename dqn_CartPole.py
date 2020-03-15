import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import statistics
import copy


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(4, 120) 
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 2)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x                  

eps_end = 0.01
eps_start = 1.0
eps_decay = 100

lr_end = 0.1
lr_start = 1.0
lr_decay = 50

critic = Net()
critic = critic.double()


actor = Net()
actor = actor.double()
optimizer = optim.SGD(actor.parameters(), lr=0.1)
criterion = nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v0')
env.reset()
time_setp = []

replay_exp = []
SIZE = 50

for eposide in range(1000):
  epsilon_ = eps_end + (eps_start - eps_end) * math.exp(-1.0 * eposide / eps_decay)
  learning_rate = lr_end + (lr_start - lr_end) * math.exp(-1.0 * eposide / lr_decay)
  observation = env.reset()
  if(eposide%100 == 0):
    critic = copy.deepcopy(actor)
  for t in range(200):
    env.render()
    #Greedy action
    output = actor(torch.from_numpy(observation))
    if random.random() < epsilon_:
      action = env.action_space.sample()
      Qvalue = output[action]
    else:
      Qvalue, action = torch.max(output,0)
      action = action.item()

    observation_next, reward, done, info = env.step(action)
    # replay_exp.append((observation, action, reward, observation_next))
    # if(len(replay_exp) > SIZE):
    #   replay_exp.pop(0)
    
    # #update
    # observation, action, reward, observation_next = replay_exp[random.randint(0, min(SIZE, len(replay_exp))-1)]
    # output = actor(torch.from_numpy(observation))
    Qvalue = output[action]
    target = output.clone()
    target[action] = Qvalue + learning_rate*(reward + 1.0 * torch.max(critic(torch.from_numpy(observation_next))) - Qvalue) 
  
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("eposide = {}\tloss = {:.3e}\tepsilon_rate = {:.2f}\tlearning_rate = {:.2f}".format(eposide, loss.item(), epsilon_, learning_rate), end ="\r")
    observation = observation_next

    if done :
      time_setp.append(t+1)
      print("Episode {} finished after {} timesteps and average = {:.2f}                           ".format(eposide, t+1, statistics.mean(time_setp)))
      break
env.close()
