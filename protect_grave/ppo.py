import os
import glob
import copy
import random 
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

tfd = tfp.distributions
tf.keras.backend.set_floatx('float32')


"""

Received data:

one sample : [5 states], [5 actions], [5 lists of prob and value], [a reward]

"""

inputs = [[[random.random() for x in range(1509)] for _ in range(5)] for i in range(2)]
# inputs = tf.constant(inputs, dtype=tf.float32)
# outputs = model(inputs)

# "slmodel/sv8feature0"
def load_model(path):
  newest = max(glob.glob(path+"/*"), key = os.path.getctime)
  return tf.saved_model.load(newest)


class MAPPO:
  def __init__(
    self,
    lr=5e-4,
    c1=1.0,
    c2=0.01,
    clip_ratio=0.2,
    gamma=0.95,
    lam=1.0,
    batch_size=128,
    n_updates=4,
  ):
    self.gamma = gamma  # discount factor
    self.lam = lam
    self.c1 = c1  # value difference coeff
    self.c2 = c2  # entropy coeff
    self.clip_ratio = clip_ratio  # for clipped surrogate
    self.batch_size = batch_size
    self.n_updates = n_updates  # number of epochs per episode
    self.model = load_model("slmodel")
    self.model_optimizer = Adam(learning_rate=lr)

  def save_model(self, path):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tf.saved_model.save(self.model, path + "/" + current_time)
  
  def squeeze(self,unsqueezed_list):
    data = []
    for row in unsqueezed_list:
      for cell in row:
        data.append(cell)
    return data

  def get_data(self):
    # obs, actions, log_probs, rewards, v_preds, next_v_preds = [], [], [], [], [], []
    obs, actions, log_probs_And_v_preds, rewards, next_v_preds = inputs, [[random.randint(0,17) for _ in range(5)] for i in range(2)], [[[random.random() for i in range(2)] for i in range(5)] for i in range(2)], [[100 for i in range(5)] for i in range(2)], [[]]
    log_probs = [ [cell[0] for cell in row] for row in log_probs_And_v_preds]
    v_preds = [ [cell[-1] for cell in row] for row in log_probs_And_v_preds]
    # next_v_preds = [0 for i in range(5)]
    next_v_preds = v_preds[1:] + [[0 for i in range(5)]]
    gaes = self.get_gaes(rewards, v_preds, next_v_preds)
    gaes = np.array(gaes).astype(dtype=np.float32)
    gaes = (gaes - gaes.mean()) / gaes.std()
    # print(len(self.squeeze(obs)))
    # print(len(self.squeeze(gaes)))
    # exit(-1)

    return [self.squeeze(obs), self.squeeze(actions), self.squeeze(log_probs), self.squeeze(next_v_preds), self.squeeze(rewards), self.squeeze(gaes)]


  def get_gaes(self, rewards, v_preds, next_v_preds):
    # source: https://github.com/uidilr/ppo_tf/blob/master/ppo.py#L98
    deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(np.array(rewards), np.array(next_v_preds), np.array(v_preds))]
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
      gaes[t] = gaes[t] + self.lam * self.gamma * gaes[t + 1]
    return gaes

  def get_dist(self, output):
    dist = tfd.Categorical(probs=output)
    return dist

  def evaluate_actions(self, states, actions):
    # output= self.model(states)
    # output, value =  [[random.random() for i in range(18)] for _ in range(5)], [random.random() for i in range(5)]
    output, value =  self.model(states), [random.random() for i in range(5)]
    dist = self.get_dist(output)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    return log_probs, entropy, value

  def learn(self, observations, actions, log_probs, next_v_preds, rewards, gaes):
    rewards = np.expand_dims(rewards, axis=-1).astype(np.float32)
    next_v_preds = np.expand_dims(next_v_preds, axis=-1).astype(np.float32)

    with tf.GradientTape() as tape:
      new_log_probs, entropy, state_values = self.evaluate_actions(observations, actions)

      ratios = tf.exp(new_log_probs - log_probs)
      clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1-self.clip_ratio,
                                        clip_value_max=1+self.clip_ratio)
      loss_clip = tf.minimum(gaes * ratios, gaes * clipped_ratios)
      loss_clip = tf.reduce_mean(loss_clip)

      target_values = rewards + self.gamma * next_v_preds
      vf_loss = tf.reduce_mean(tf.math.square(state_values - target_values))
      vf_loss = tf.dtypes.cast(vf_loss, tf.float32)
      entropy = tf.reduce_mean(entropy)

      total_loss = -loss_clip + self.c1 * vf_loss - self.c2 * entropy
      # total_loss = -loss_clip

    train_variables = self.model.trainable_variables
    # f = open("before_bp.txt", "w")
    # print(train_variables, file=f)
    grad = tape.gradient(total_loss, train_variables)  # compute gradient
    self.model_optimizer.apply_gradients(zip(grad, train_variables))
    # f = open("after_bp.txt", "w")
    # print(train_variables, file=f)




if __name__ == "__main__":
  mappo = MAPPO()
  data = mappo.get_data()
  # print(*data)
  mappo.learn(*data)
  mappo.save_model("slmodel")
  f = open("write.txt", "w")
  print(mappo.model.trainable_variables, file=f)

