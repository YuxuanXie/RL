import gym
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from torch.autograd import Variable
import statistics


# hyper parameters
EPISODES = 200  # number of episodes
GAMMA = 0.95  # Q-learning discount factor
LR = 0.01  # NN optimizer learning rate
HIDDEN_LAYER = 48  # NN hidden layer size
BATCH_SIZE = 16 # Q-learning batch size

eps_end = 0.01
eps_start = 1
eps_decay = 50

lr_end = 0.01
lr_start = 1.0
lr_decay = 100

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class ReplayMemory:
  def __init__(self, cap):
    self.cap = cap
    self.memory = []
  
  def push(self, m):
    self.memory.append(m)
    if(len(memory) > self.cap):
      del self.memory[0]

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)
  
  def __len__(self):
    return len(self.memory)

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.l1 = nn.Linear(4, HIDDEN_LAYER)
    self.l2 = nn.Linear(HIDDEN_LAYER, 2)

  def forward(self, x):
    x = torch.tanh(self.l1(x))
    x = self.l2(x)
    return x

  def fit(self, input, expected_output):
    for i in range(10):
      loss = F.mse_loss(self.forward(input), expected_output)
      optimizer.zero_grad()
      loss.backward(retain_graph=True)
      optimizer.step()
    return loss



env = gym.make('CartPole-v0')
model = Network()
model = model.double()


memory = ReplayMemory(1000)
optimizer = optim.Adagrad(model.parameters(), LR)
steps_done = 0
time_setp = []


def select_action(state, epsilon_threshold):
  global steps_done
  sample = random.random()
  steps_done += 1
  if sample < epsilon_threshold:
    return LongTensor([[random.randrange(2)]])
  else:
    return model(Variable(state).type(FloatTensor)).data.max(1)[1].view(1,1)


def run_eposide(e, environment, learning_rate):
  global eps_start
  state = environment.reset()
  steps = 0
  while True:
    environment.render()
    action = select_action(FloatTensor([state]), eps_start)
    state_next, reward, done, info = environment.step(action.data[0,0].item())

    memory.push([FloatTensor([state]), action, FloatTensor([reward]), FloatTensor([state_next]), FloatTensor([done])])

    loss = learn(learning_rate)
    loss = 0 if(loss == None) else loss

    steps+=1
    eps_start *= 0.995

    state = state_next

    if(done):
      time_setp.append(steps)
      print("{2} Episode {0} steps = {1} lr-rate = {3:.2f} explore-rate = {4:.2f} avg = {5:.2f} loss = {6:.2f}".format(e, steps, '\033[92m' if steps >= 195 else '\033[99m', learning_rate, eps_start, statistics.mean(time_setp), loss))
      break

def learn(learning_rate):
  if len(memory) < BATCH_SIZE:
    return
  
  transitions = memory.sample(BATCH_SIZE)

  for state, action, reward, next_state, done in transitions:
    if done :
      target = reward
    else:
      target = reward + GAMMA * model(next_state).max(1)[0]
    
    current_q_values = model(state)

    target_q_values = current_q_values.clone()
    target_q_values[0,action] = target 

    loss = model.fit(state, target_q_values)

    # loss = F.mse_loss(current_q_values, target_q_values)
    
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()


  # batch_state, batch_action, batch_reward, batch_next_state, done = zip(*transitions)

  # batch_state = Variable(torch.cat(batch_state))
  # batch_action = Variable(torch.cat(batch_action))
  # batch_reward = Variable(torch.cat(batch_reward)).view(BATCH_SIZE,1)
  # batch_next_state = Variable(torch.cat(batch_next_state))
  # batch_done = 1 - Variable(torch.cat(done)).view(BATCH_SIZE,1)


  # current_q_values = model(batch_state).gather(1,batch_action)
  # max_next_q_values = model(batch_next_state).max(1)[0].view(BATCH_SIZE,1)
  # expected_q_values = batch_reward + GAMMA * max_next_q_values.mul(batch_done) 
  # # error = learning_rate*(batch_reward + GAMMA * max_next_q_values  - current_q_values)
  # # expected_q_values = current_q_values +  error.mul(batch_done)

  # # print(batch_reward.view(BATCH_SIZE,1))
  # # print("------current_q_values--------")
  # # print(current_q_values)
  # # print("-------max_next_q_values-------")
  # # print(max_next_q_values)
  # # print("---------expected_q_values-----")
  # # print(expected_q_values)
  # # print("---------batch_done-----")
  # # print(batch_done.view(BATCH_SIZE,1))
  # # exit(0)

  # optimizer.zero_grad()
  # loss = F.nll_loss(current_q_values, expected_q_values)
  # loss.backward()
  # optimizer.step()

  return loss

for e in range(EPISODES):
  # epsilon_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * e / eps_decay)
  learning_rate = lr_end + (lr_start - lr_end) * math.exp(-1.0 * e / lr_decay)
  run_eposide(e, env, learning_rate)

# print('Complete')
env.close()



