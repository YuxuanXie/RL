import gym
import copy
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import models,layers
from tensorflow.keras.optimizers import Adam

tfd = tfp.distributions
env = gym.make("CartPole-v1")
vector_length = env.observation_space.shape[0]
tf.keras.backend.set_floatx('float64')
action_length = env.action_space.n


class MAPPO:
  def __init__(
    self,
    lr=5e-4,
    hidden_units=(24, 16),
    c1=1.0,
    c2=0.01,
    clip_ratio=0.2,
    gamma=0.95,
    lam=1.0,
    batch_size=64,
    n_updates=4,
  ):
    self.gamma = gamma  # discount factor
    self.lam = lam
    self.c1 = c1  # value difference coeff
    self.c2 = c2  # entropy coeff
    self.clip_ratio = clip_ratio  # for clipped surrogate
    self.batch_size = batch_size
    self.n_updates = n_updates  # number of epochs per episode
    self.model_optimizer = Adam(learning_rate=lr)

    self.actor_model = models.Sequential()
    self.actor_model.add(layers.Dense(4, activation="tanh", input_shape=(vector_length,)))
    self.actor_model.add(layers.Dense(400, activation="tanh"))
    self.actor_model.add(layers.Dense(300, activation="tanh"))
    self.actor_model.add(layers.Dense(100, activation="tanh"))
    self.actor_model.add(layers.Dense(action_length, activation="softmax"))


    self.critic_model = models.Sequential()
    self.critic_model.add(layers.Dense(4, activation="tanh", input_shape=(vector_length,)))
    self.critic_model.add(layers.Dense(400, activation="tanh"))
    self.critic_model.add(layers.Dense(300, activation="tanh"))
    self.critic_model.add(layers.Dense(100, activation="tanh"))
    self.critic_model.add(layers.Dense(1))

    self.summaries = {}


  def get_dist(self, output):
    dist = tfd.Categorical(probs=output)
    return dist

  def evaluate_actions(self, states, actions):
    # states = tf.constant(states)
    # print(states)
    output = self.actor_model(states)
    value = self.critic(states)
    dist = self.get_dist(output)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    return log_probs, entropy, value

  def actor(self, state):
    if len(state.shape) == 1:
      state = np.expand_dims(state, axis=0).astype(np.float64)
    output = self.actor_model.predict(state)
    dist = self.get_dist(output)

    action = dist.sample()
    log_probs = dist.log_prob(action)

    return action[0].numpy(), log_probs[0].numpy()


  def critic(self, state):
    if len(state.shape) == 1:
      state = np.expand_dims(state, axis=0).astype(np.float64)
    return self.critic_model(state)

  def get_gaes(self, rewards, v_preds, next_v_preds):
    # source: https://github.com/uidilr/ppo_tf/blob/master/ppo.py#L98
    deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, next_v_preds, v_preds)]
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
      gaes[t] = gaes[t] + self.lam * self.gamma * gaes[t + 1]
    return gaes

  def save_model(self, fn):
    self.actor_model.save(fn+"_actor.h5")
    self.critic_model.save(fn+"_critic.h5")

  def learn(self, observations, actions, log_probs, next_v_preds, rewards, gaes):
    rewards = np.expand_dims(rewards, axis=-1).astype(np.float64)
    next_v_preds = np.expand_dims(next_v_preds, axis=-1).astype(np.float64)

    with tf.GradientTape(persistent=True) as tape:
      new_log_probs, entropy, state_values = self.evaluate_actions(observations, actions)

      ratios = tf.exp(new_log_probs - log_probs)
      clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1-self.clip_ratio,
                                        clip_value_max=1+self.clip_ratio)
      loss_clip = tf.minimum(gaes * ratios, gaes * clipped_ratios)
      loss_clip = tf.reduce_mean(loss_clip)

      target_values = rewards + self.gamma * next_v_preds
      vf_loss = tf.reduce_mean(tf.math.square(state_values - target_values))

      entropy = tf.reduce_mean(entropy)
      actor_loss = -loss_clip - self.c2 * entropy
    
    train_actor_variables = self.actor_model.trainable_variables
    train_critic_variables = self.critic_model.trainable_variables

    grad = tape.gradient(actor_loss, train_actor_variables)  # compute gradient
    self.model_optimizer.apply_gradients(zip(grad, train_actor_variables))

    grad = tape.gradient(vf_loss, train_critic_variables)  # compute gradient
    self.model_optimizer.apply_gradients(zip(grad, train_critic_variables))

    # tensorboard info
    self.summaries['actor_loss'] = actor_loss
    self.summaries['surr_loss'] = loss_clip
    self.summaries['vf_loss'] = vf_loss
    self.summaries['entropy'] = entropy


  def train(self, max_epochs=8000, max_steps=500, save_freq=50):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/' + current_time
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    episode, epoch = 0, 0

    while epoch < max_epochs:
      done, steps = False, 0
      cur_state = env.reset()
      obs, actions, log_probs, rewards, v_preds, next_v_preds = [], [], [], [], [], []

      while not done and steps < max_steps:
        action, log_prob = self.actor(cur_state)  # determine action
        value = self.critic(cur_state)
        next_state, reward, done, _ = env.step(action)  # act on env
        # self.env.render(mode='rgb_array')

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
      gaes = (gaes - gaes.mean()) / gaes.std()
      data = [obs, actions, log_probs, next_v_preds, rewards, gaes]

      for i in range(self.n_updates):
        # Sample training data
        sample_indices = np.random.randint(low=0, high=len(rewards), size=self.batch_size)
        sampled_data = [np.take(a=a, indices=sample_indices, axis=0) for a in data]

        # # Train model
        self.learn(*sampled_data)

        # Tensorboard update
        with summary_writer.as_default():
          tf.summary.scalar('Loss/actor_loss', self.summaries['actor_loss'], step=epoch)
          tf.summary.scalar('Loss/clipped_surr', self.summaries['surr_loss'], step=epoch)
          tf.summary.scalar('Loss/vf_loss', self.summaries['vf_loss'], step=epoch)
          tf.summary.scalar('Loss/entropy', self.summaries['entropy'], step=epoch)

        summary_writer.flush()
        epoch += 1

      episode += 1
      print("episode {}: {} total reward, {} steps, {} epochs".format(
        episode, np.sum(rewards), steps, epoch))

      # Tensorboard update
      with summary_writer.as_default():
        tf.summary.scalar('Main/episode_reward', np.sum(rewards), step=episode)
        tf.summary.scalar('Main/episode_steps', steps, step=episode)

      summary_writer.flush()

      if steps >= max_steps:
        print("episode {}, reached max steps".format(episode))
        self.save_model("yxppo_episode{}.h5".format(episode))

      if episode % save_freq == 0:
        self.save_model("yxppo_episode{}".format(episode))

    self.save_model("yxppo_final_episode{}".format(episode))


if __name__ == "__main__":
  ppo = MAPPO()
  print(ppo.actor_model.summary())
  # print(ppo.critic_model.summary())
  ppo.train(max_epochs=1000, save_freq=190)
