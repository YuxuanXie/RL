"""
PyTorch分布式DQN训练 - 使用DDP (DistributedDataParallel)
支持多个K8s pod进行分布式训练
"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import math
import statistics
import os


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


def setup_distributed():
    """初始化分布式训练环境"""
    # 从环境变量获取分布式训练参数
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '23456')
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # 初始化进程组
    dist.init_process_group(
        backend='gloo',  # 使用gloo后端（CPU）或'nccl'（GPU）
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return rank, world_size


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_distributed():
    """分布式训练主函数"""
    # 设置分布式环境
    rank, world_size = setup_distributed()
    
    print(f"[Rank {rank}] 初始化成功，共 {world_size} 个worker")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型并包装为DDP
    actor = Net().double().to(device)
    actor = DDP(actor)
    
    critic = Net().double().to(device)
    
    optimizer = optim.SGD(actor.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    # 每个worker创建自己的环境（不同的随机种子）
    env = gym.make('CartPole-v1')
    env.seed(rank)  # 不同worker使用不同种子
    random.seed(rank)
    torch.manual_seed(rank)
    
    eps_end = 0.01
    eps_start = 1.0
    eps_decay = 100
    
    lr_end = 0.1
    lr_start = 1.0
    lr_decay = 50
    
    time_steps = []
    
    # 只有rank 0保存结果
    if rank == 0:
        f = open(f"result_distributed_rank{rank}.csv", "w")
        f.write("timesteps,trial\n")
    
    def select_action(observation, epsilon):
        with torch.no_grad():
            output = actor(torch.from_numpy(observation).to(device))
        if random.random() < epsilon:
            action = env.action_space.sample()
            Qvalue = output[action]
        else:
            Qvalue, action = torch.max(output, 0)
            action = action.item()
        return output, action
    
    for episode in range(300):
        epsilon_ = eps_end + (eps_start - eps_end) * math.exp(-1.0 * episode / eps_decay)
        learning_rate = lr_end + (lr_start - lr_end) * math.exp(-1.0 * episode / lr_decay)
        
        observation = env.reset()
        
        for t in range(200):
            # Greedy action
            output, action = select_action(observation, epsilon_)
            
            observation_next, reward, done, info = env.step(action)
            
            Qvalue = output[action]
            target = output.clone()
            with torch.no_grad():
                next_q = torch.max(actor(torch.from_numpy(observation_next).to(device)))
            target[action] = Qvalue + learning_rate * (reward + 1.0 * next_q - Qvalue)
            
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            
            # DDP会自动同步梯度
            optimizer.step()
            
            if rank == 0 and t % 50 == 0:
                print(f"[Rank {rank}] episode={episode}\tloss={loss.item():.3e}\tepsilon={epsilon_:.2f}\tlr={learning_rate:.2f}", end="\r")
            
            observation = observation_next
            
            if done:
                time_steps.append(t + 1)
                if rank == 0:
                    f.write(f"{t+1},{episode}\n")
                    avg = statistics.mean(time_steps) if time_steps else 0
                    print(f"[Rank {rank}] Episode {episode} finished after {t+1} timesteps, avg={avg:.2f}           ")
                break
        
        # 每10个episode同步一次统计信息
        if episode % 10 == 0:
            # 收集所有worker的平均reward
            if len(time_steps) > 0:
                local_avg = torch.tensor(statistics.mean(time_steps), device=device)
            else:
                local_avg = torch.tensor(0.0, device=device)
            
            # 全局平均
            dist.all_reduce(local_avg, op=dist.ReduceOp.SUM)
            global_avg = local_avg.item() / world_size
            
            if rank == 0:
                print(f"\n[Rank {rank}] Episode {episode}: 全局平均时间步 = {global_avg:.2f}\n")
    
    if rank == 0:
        f.close()
        # 保存模型
        torch.save(actor.module.state_dict(), "dqn_distributed_model.pt")
        print(f"[Rank {rank}] 训练完成，模型已保存")
    
    env.close()
    cleanup_distributed()


if __name__ == "__main__":
    train_distributed()
