import gym
import numpy as np
import math
from ppo import PPO
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir=f"./log/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

def train(alg, max_epochs=8000, max_steps=500, save_freq=50):

    episode, epoch = 0, 0
    while epoch < max_epochs:
        done, steps = False, 0
        cur_state = env.reset()
        obs, actions, log_probs, rewards, v_preds, next_v_preds = [], [], [], [], [], []

        while not done and steps < max_steps:
            action, value, log_prob = alg.act({1 : cur_state})  # determine action
            action, value, log_prob = [ list(each.values())[0] for each in [action, value, log_prob]]
            next_state, reward, done, _ = env.step(action[0].item())  # act on env
            # self.env.render(mode='rgb_array')

            rewards.append(reward)
            v_preds.append(value.detach())
            obs.append(cur_state)
            actions.append(action)
            log_probs.append(log_prob)

            steps += 1
            cur_state = next_state

        next_v_preds = v_preds[1:] + [0]
        gaes = alg.get_gaes(rewards, v_preds, next_v_preds)
        gaes = np.array(gaes).astype(dtype=np.float64)
        gaes = (gaes - gaes.mean()) / gaes.std()
        data = [obs, actions, log_probs, next_v_preds, rewards, gaes]


        if len(data[-2]) > 0 :
            writer.add_scalar("Info/mean_reward", steps, alg.update_steps)
            writer.add_scalar("Info/max_pi", math.exp(max(data[2])), alg.update_steps)

        for i in range(alg.n_update):
            # Sample training data
            sample_indices = np.random.randint(low=0, high=len(rewards), size=alg.batch_size)
            sampled_data = [np.take(a=a, indices=sample_indices, axis=0) for a in data]
            # # Train model
            loss, policy_loss, value_loss, entropy = alg.learn(*sampled_data)
            if i == 0:
                writer.add_scalar("Info/total_loss", loss.item(), alg.update_steps)
                writer.add_scalar("Info/policy_loss", policy_loss.item(), alg.update_steps)
                writer.add_scalar("Info/value_loss", value_loss.item(), alg.update_steps)
                writer.add_scalar("Info/entropy", entropy.item(), alg.update_steps)
        epoch += 1
        episode += 1
        print("episode {}: {} total reward, {} steps, {} epochs".format(episode, np.sum(rewards), steps, epoch))


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # Gamma and lambda matter!!!
    alg = PPO(state_size, [action_size], n_updates=4, learning_rate=5e-4, clip_ratio=0.1, hidden_size=32, c2=0.01, gamma=0.99, lam=1.0)

    train(alg)

