# Kubernetes 多机训练配置指南

本指南介绍如何在K8s集群中使用2个pod进行分布式强化学习训练。

## 📋 目录

1. [方案概述](#方案概述)
2. [PyTorch分布式训练](#pytorch分布式训练)
3. [TensorFlow分布式训练](#tensorflow分布式训练)
4. [常见问题](#常见问题)

---

## 方案概述

我们提供了两种主流的分布式训练方案：

| 方案 | 框架 | 适用算法 | 通信后端 |
|------|------|----------|----------|
| **PyTorch DDP** | PyTorch | DQN等 | Gloo/NCCL |
| **TF MultiWorkerMirroredStrategy** | TensorFlow | PPO等 | gRPC |

### 架构说明

```
┌─────────────────────────────────────────┐
│         Kubernetes Cluster              │
│                                         │
│  ┌─────────────┐    ┌─────────────┐   │
│  │   Pod 0     │    │   Pod 1     │   │
│  │  (Master)   │◄──►│  (Worker)   │   │
│  │             │    │             │   │
│  │  Rank 0     │    │  Rank 1     │   │
│  └─────────────┘    └─────────────┘   │
│         ▲                  ▲           │
│         └──────────────────┘           │
│         Headless Service               │
└─────────────────────────────────────────┘
```

---

## PyTorch分布式训练

### 1. 部署步骤

#### Step 1: 应用K8s配置

```bash
# 创建PyTorch分布式训练任务
kubectl apply -f k8s/pytorch-distributed-job.yaml

# 查看pod状态
kubectl get pods -l app=pytorch-training

# 查看日志
kubectl logs -f pytorch-training-0
kubectl logs -f pytorch-training-1
```

#### Step 2: 验证训练启动

```bash
# 查看所有pod的日志
kubectl logs pytorch-training-0 --tail=50
kubectl logs pytorch-training-1 --tail=50
```

你应该能看到类似输出：
```
[Rank 0] 初始化成功，共 2 个worker
[Rank 0] episode=0	loss=1.234e-01	epsilon=1.00	lr=1.00
[Rank 0] Episode 0 finished after 15 timesteps, avg=15.00
```

### 2. 配置说明

#### 环境变量（自动设置）

```bash
RANK=0                    # Pod序号，0为master
WORLD_SIZE=2              # 总worker数量
MASTER_ADDR=pytorch-training-0.pytorch-distributed-svc.default.svc.cluster.local
MASTER_PORT=23456         # 通信端口
```

#### 修改副本数

在 `pytorch-distributed-job.yaml` 中修改：

```yaml
spec:
  replicas: 3  # 改为3个pod
```

对应修改代码中的 `WORLD_SIZE`:

```bash
export WORLD_SIZE=3
```

### 3. 核心代码解析

```python
# 初始化分布式
dist.init_process_group(
    backend='gloo',        # CPU用gloo，GPU用nccl
    init_method='env://',  # 从环境变量读取配置
    world_size=world_size,
    rank=rank
)

# 模型包装为DDP
model = DDP(model)

# DDP会自动同步梯度
optimizer.step()  # 所有worker的梯度会自动平均
```

### 4. 性能优化

#### GPU加速

如果有GPU，修改配置：

```yaml
# pytorch-distributed-job.yaml
command: ["/bin/bash", "-c"]
args:
  - |
    export NCCL_SOCKET_IFNAME=eth0  # 网络接口
    export NCCL_DEBUG=INFO          # 调试信息
    python /workspace/CartPole/DQN/dqn_distributed.py
```

代码中使用NCCL后端：

```python
dist.init_process_group(backend='nccl')  # GPU通信更快
```

---

## TensorFlow分布式训练

### 1. 部署步骤

#### Step 1: 应用K8s配置

```bash
# 创建TensorFlow分布式训练任务
kubectl apply -f k8s/tensorflow-distributed-job.yaml

# 查看pod状态
kubectl get pods -l app=tf-training

# 查看日志
kubectl logs -f tf-training-0
kubectl logs -f tf-training-1
```

#### Step 2: 验证TF_CONFIG

```bash
# 查看worker 0的TF_CONFIG
kubectl exec tf-training-0 -- printenv TF_CONFIG
```

应该看到：
```json
{
  "cluster": {
    "worker": [
      "tf-training-0.tf-distributed-svc.default.svc.cluster.local:12345",
      "tf-training-1.tf-distributed-svc.default.svc.cluster.local:12345"
    ]
  },
  "task": {
    "type": "worker",
    "index": 0
  }
}
```

### 2. 配置说明

#### TF_CONFIG环境变量

TensorFlow通过 `TF_CONFIG` 环境变量配置分布式：

```json
{
  "cluster": {
    "worker": ["host1:port1", "host2:port2"]  // 所有worker地址
  },
  "task": {
    "type": "worker",     // 任务类型
    "index": 0            // 当前worker的索引
  }
}
```

#### 修改worker数量

在 `tensorflow-distributed-job.yaml` 中：

```yaml
spec:
  replicas: 3  # 改为3个worker

args:
  - |
    export TF_CONFIG=$(cat <<EOF
    {
      "cluster": {
        "worker": [
          "tf-training-0.tf-distributed-svc:12345",
          "tf-training-1.tf-distributed-svc:12345",
          "tf-training-2.tf-distributed-svc:12345"  # 添加第3个
        ]
      },
      "task": {
        "type": "worker",
        "index": ${WORKER_INDEX}
      }
    }
    EOF
    )
```

### 3. 核心代码解析

```python
# 创建分布式策略
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# 在strategy scope中创建模型
with strategy.scope():
    model = create_model()
    optimizer = Adam(learning_rate=lr)

# 使用@tf.function装饰训练步骤
@tf.function
def train_step(data):
    # 训练逻辑
    pass

# 策略会自动处理分布式
```

### 4. 通信优化

#### 配置通信选项

在pod启动脚本中添加：

```bash
# 使用更高效的通信协议
export TF_COMMUNICATION_OPTIONS="nccl"

# 配置gRPC选项
export GRPC_VERBOSITY=DEBUG
export GRPC_TRACE=all
```

---

## 常见问题

### Q1: Pod无法互相通信？

**解决方案：**

1. 检查Headless Service是否创建成功：
```bash
kubectl get svc pytorch-distributed-svc
kubectl get svc tf-distributed-svc
```

2. 检查DNS解析：
```bash
kubectl exec pytorch-training-0 -- nslookup pytorch-training-1.pytorch-distributed-svc
```

3. 检查端口是否开放：
```bash
kubectl exec pytorch-training-0 -- nc -zv pytorch-training-1.pytorch-distributed-svc 23456
```

### Q2: 训练速度没有提升？

**可能原因：**

1. **通信开销大于计算收益**
   - CartPole环境太简单，模型太小
   - 建议：使用更复杂的环境（如Atari）

2. **网络带宽不足**
   - 检查节点间网络：
   ```bash
   kubectl exec pytorch-training-0 -- iperf3 -c pytorch-training-1
   ```

3. **负载不均衡**
   - 确保每个worker处理相同的数据量

### Q3: 如何监控训练进度？

**TensorBoard（TensorFlow）：**

```bash
# 启动TensorBoard服务
kubectl port-forward tf-training-0 6006:6006

# 在浏览器访问
http://localhost:6006
```

**自定义监控：**

```bash
# 查看实时日志
kubectl logs -f pytorch-training-0 | grep "Episode"

# 导出结果文件
kubectl cp pytorch-training-0:/workspace/result_distributed_rank0.csv ./results.csv
```

### Q4: 如何保存和加载模型？

**使用PVC（推荐）：**

```yaml
# 在yaml中配置PVC
volumes:
- name: model-storage
  persistentVolumeClaim:
    claimName: training-model-pvc

volumeMounts:
- name: model-storage
  mountPath: /models
```

**从pod中复制：**

```bash
# PyTorch
kubectl cp pytorch-training-0:/workspace/dqn_distributed_model.pt ./model.pt

# TensorFlow
kubectl cp tf-training-0:/workspace/ppo_distributed_final_episode100.h5 ./model.h5
```

### Q5: 训练中断如何恢复？

**添加checkpoint机制：**

```python
# PyTorch
if episode % 100 == 0 and rank == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'checkpoint_ep{episode}.pt')

# TensorFlow
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint.save('/models/ckpt')
```

### Q6: 如何调试分布式训练？

**启用详细日志：**

```bash
# PyTorch
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# TensorFlow  
export TF_CPP_MIN_LOG_LEVEL=0
export GRPC_VERBOSITY=DEBUG
```

**进入pod交互式调试：**

```bash
kubectl exec -it pytorch-training-0 -- /bin/bash

# 手动运行训练脚本
python /workspace/CartPole/DQN/dqn_distributed.py
```

---

## 进阶配置

### 混合精度训练（提速）

**PyTorch:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**TensorFlow:**
```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### 梯度累积（节省显存）

```python
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 动态调整worker数量

```bash
# 扩容
kubectl scale statefulset pytorch-training --replicas=4

# 缩容
kubectl scale statefulset pytorch-training --replicas=2
```

---

## 清理资源

```bash
# 删除PyTorch训练任务
kubectl delete -f k8s/pytorch-distributed-job.yaml

# 删除TensorFlow训练任务
kubectl delete -f k8s/tensorflow-distributed-job.yaml

# 查看资源是否清理完成
kubectl get all -l app=pytorch-training
kubectl get all -l app=tf-training
```

---

## 参考资料

- [PyTorch Distributed](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [TensorFlow Distributed](https://www.tensorflow.org/guide/distributed_training)
- [Kubernetes StatefulSet](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [Horovod](https://github.com/horovod/horovod) - 另一个分布式训练框架

---

## 联系与支持

如有问题，请查看：
- Pod日志: `kubectl logs <pod-name>`
- 事件: `kubectl describe pod <pod-name>`
- 集群状态: `kubectl get nodes`
