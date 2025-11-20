"""
TensorFlow分布式PPO训练 - 使用MultiWorkerMirroredStrategy
支持多个K8s pod进行分布式训练
"""
import gym
import copy
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
import json
import os

tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')


def model(state_shape, action_dim, units=(400, 300, 100)):
    state = Input(shape=state_shape)

    vf = Dense(units[0], name="Value_L0", activation="tanh")(state)
    for index in range(1, len(units)):
        vf = Dense(units[index], name="Value_L{}".format(index), activation="tanh")(vf)

    value_pred = Dense(1, name="Out_value")(vf)

    pi = Dense(units[0], name="Policy_L0", activation="tanh")(state)
    for index in range(1, len(units)):
        pi = Dense(units[index], name="Policy_L{}".format(index), activation="tanh")(pi)

    action_probs = Dense(action_dim, name="Out_probs", activation='softmax')(pi)
    model = Model(inputs=state, outputs=[action_probs, value_pred])

    return model


class DistributedMAPPO:
    def __init__(
        self,
        strategy,
        vector_length,
        action_length,
        lr=5e-4,
        hidden_units=(24, 16),
        c1=1.0,
        c2=0.01,
        clip_ratio=0.2,
        gamma=0.95,
        lam=1.0,
        batch_size=128,
        n_updates=4,
    ):
        self.gamma = gamma
        self.lam = lam
        self.c1 = c1
        self.c2 = c2
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.n_updates = n_updates
        self.strategy = strategy
        
        # 在strategy scope中创建模型和优化器
        with strategy.scope():
            self.model = model(vector_length, action_length, hidden_units)
            self.model_optimizer = Adam(learning_rate=lr)
        
        self.summaries = {}

    def get_dist(self, output):
        dist = tfd.Categorical(probs=output)
        return dist

    def evaluate_actions(self, states, actions):
        output, value = self.model(states)
        dist = self.get_dist(output)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value

    def act(self, state):
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0).astype(np.float64)
        output, value = self.model(state)
        dist = self.get_dist(output)

        action = dist.sample()
        log_probs = dist.log_prob(action)
        return action[0].numpy(), value[0][0], log_probs[0].numpy()

    def get_gaes(self, rewards, v_preds, next_v_preds):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, next_v_preds, v_preds)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.lam * self.gamma * gaes[t + 1]
        return gaes

    def save_model(self, fn):
        self.model.save(fn + ".h5")

    @tf.function
    def train_step(self, observations, actions, log_probs, next_v_preds, rewards, gaes):
        """分布式训练步骤"""
        with tf.GradientTape() as tape:
            new_log_probs, entropy, state_values = self.evaluate_actions(observations, actions)
            ratios = tf.exp(new_log_probs - log_probs)
            clipped_ratios = tf.clip_by_value(
                ratios, 
                clip_value_min=1 - self.clip_ratio,
                clip_value_max=1 + self.clip_ratio
            )
            loss_clip = tf.minimum(gaes * ratios, gaes * clipped_ratios)
            loss_clip = tf.reduce_mean(loss_clip)

            target_values = rewards + self.gamma * next_v_preds
            vf_loss = tf.reduce_mean(tf.math.square(state_values - target_values))

            entropy = tf.reduce_mean(entropy)
            total_loss = -loss_clip + self.c1 * vf_loss - self.c2 * entropy

        train_variables = self.model.trainable_variables
        grad = tape.gradient(total_loss, train_variables)
        self.model_optimizer.apply_gradients(zip(grad, train_variables))

        return total_loss, loss_clip, vf_loss, entropy

    def learn(self, observations, actions, log_probs, next_v_preds, rewards, gaes):
        """包装训练步骤以支持分布式"""
        rewards = np.expand_dims(rewards, axis=-1).astype(np.float64)
        next_v_preds = np.expand_dims(next_v_preds, axis=-1).astype(np.float64)

        # strategy.run会自动处理分布式
        total_loss, loss_clip, vf_loss, entropy = self.train_step(
            observations, actions, log_probs, next_v_preds, rewards, gaes
        )

        self.summaries['total_loss'] = total_loss
        self.summaries['surr_loss'] = loss_clip
        self.summaries['vf_loss'] = vf_loss
        self.summaries['entropy'] = entropy

    def train(self, env, max_epochs=8000, max_steps=500, save_freq=50):
        # 获取worker信息
        task_type = self.strategy.cluster_resolver.task_type
        task_id = self.strategy.cluster_resolver.task_id
        
        print(f"[Worker {task_id}] 开始训练，任务类型: {task_type}")
        
        # 只在worker 0上创建tensorboard
        if task_id == 0:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = f'logs/distributed_{current_time}'
            summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        episode, epoch = 0, 0

        while epoch < max_epochs:
            done, steps = False, 0
            cur_state = env.reset()
            obs, actions, log_probs, rewards, v_preds, next_v_preds = [], [], [], [], [], []

            while not done and steps < max_steps:
                action, value, log_prob = self.act(cur_state)
                next_state, reward, done, _ = env.step(action)

                rewards.append(reward)
                v_preds.append(value)
                obs.append(cur_state)
                actions.append(action)
                log_probs.append(log_prob)

                steps += 1
                cur_state = next_state

            next_v_preds = v_preds[1:] + [0]
            gaes = self.get_gaes(rewards, v_preds, next_v_preds)
            gaes = np.array(gaes).astype(dtype=np.float64)
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
            data = [obs, actions, log_probs, next_v_preds, rewards, gaes]

            for i in range(self.n_updates):
                # Sample training data
                sample_indices = np.random.randint(low=0, high=len(rewards), size=self.batch_size)
                sampled_data = [np.take(a=a, indices=sample_indices, axis=0) for a in data]

                # Train model
                self.learn(*sampled_data)

                # Tensorboard update (only worker 0)
                if task_id == 0:
                    with summary_writer.as_default():
                        tf.summary.scalar('Loss/total_loss', self.summaries['total_loss'], step=epoch)
                        tf.summary.scalar('Loss/clipped_surr', self.summaries['surr_loss'], step=epoch)
                        tf.summary.scalar('Loss/vf_loss', self.summaries['vf_loss'], step=epoch)
                        tf.summary.scalar('Loss/entropy', self.summaries['entropy'], step=epoch)
                    summary_writer.flush()

                epoch += 1

            episode += 1
            
            if task_id == 0:
                print(f"[Worker {task_id}] episode {episode}: {np.sum(rewards)} total reward, {steps} steps, {epoch} epochs")

                # Tensorboard update
                with summary_writer.as_default():
                    tf.summary.scalar('Main/episode_reward', np.sum(rewards), step=episode)
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)
                summary_writer.flush()

        # 只在worker 0上保存模型
        if task_id == 0:
            self.save_model(f"ppo_distributed_final_episode{episode}")
            print(f"[Worker {task_id}] 训练完成，模型已保存")


def setup_distributed_strategy():
    """设置TensorFlow分布式策略"""
    # 从环境变量读取TF_CONFIG
    tf_config = os.environ.get('TF_CONFIG')
    
    if tf_config:
        tf_config = json.loads(tf_config)
        print("TF_CONFIG:", json.dumps(tf_config, indent=2))
    else:
        print("未找到TF_CONFIG，使用单机模式")
        return tf.distribute.get_strategy()  # 默认策略
    
    # 使用MultiWorkerMirroredStrategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    return strategy


if __name__ == "__main__":
    # 设置分布式策略
    strategy = setup_distributed_strategy()
    
    print(f"使用策略: {strategy}")
    print(f"设备数量: {strategy.num_replicas_in_sync}")
    
    # 创建环境
    env = gym.make("CartPole-v1")
    vector_length = env.observation_space.shape[0]
    action_length = env.action_space.n
    
    # 创建分布式PPO
    ppo = DistributedMAPPO(
        strategy=strategy,
        vector_length=vector_length,
        action_length=action_length,
        n_updates=4,
        hidden_units=(24, 16)
    )
    
    print(ppo.model.summary())
    
    # 开始训练
    ppo.train(env, max_epochs=10000, save_freq=2000)
    
    env.close()
