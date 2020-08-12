import os
import glob
import copy
import pickle
import random 
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

tfd = tfp.distributions
tf.keras.backend.set_floatx('float32')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import lz4.frame
# import msgpack
# import msgpack_numpy

# msgpack_numpy.patch()

"""

Received data:

one sample : [5 states], [5 actions], [5 lists of prob and value], [5 reward]

"""

inputs = [[[random.random() for x in range(1509)] for _ in range(5)] for i in range(1000)]

def load_model(path):
  newest = max(glob.glob(path+"/*"), key = os.path.getctime)
  print(newest)
  return tf.saved_model.load(newest)



class MAPPO:
  def __init__(
    self,
    lr=5e-4,
    c1=1.0,
    c2=0.01,
    clip_ratio=0.1,
    gamma=0.999,
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
    self.model = load_model("model")
    self.model_optimizer = Adam(learning_rate=lr)


  # def _data_src(self):
  #   _f = pickle.load(open("sample.pkl", "rb"))
  #   for each in _f:
  #     decompressed = lz4.frame.decompress(each)
  #     yield msgpack.unpackb(decompressed)

  def save_model(self, path):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tf.saved_model.save(self.model, path + "/" + current_time)
  
  def squeeze(self,unsqueezed_list):
    data = []
    for d1 in unsqueezed_list:
      for d2 in d1:
        for d3 in d2:
          data.append(d3)
    return data

  def get_data(self, eposides):
    # eposides = self._data_src()
    # dict_keys([b'state', b'action', b'reward', b'done', b'probs', b'values'])
    obs, actions, probs, v_preds, rewards, next_v_preds, gaes_final = [],[],[],[],[],[],[]
    for e in eposides:
      obs.append(e[b"state"])
      actions.append(e[b"action"])
      probs.append(e[b"probs"])
      rewards_temp = e[b"reward"]
      rewards.append(rewards_temp)
      v_preds = e[b'values']

      next_v_preds_temp = v_preds[1:] + [[0 for i in range(5)]]
      next_v_preds.append(next_v_preds_temp)
      gaes = self.get_gaes(rewards_temp, v_preds, next_v_preds_temp)
      gaes = np.array(gaes).astype(dtype=np.float32)
      gaes = (gaes - gaes.mean()) / gaes.std()
      gaes_final.append(gaes)

    log_probs = self.get_dist(self.squeeze(probs)).log_prob(self.squeeze(actions))
    
    return [self.squeeze(obs), self.squeeze(actions), log_probs, self.squeeze(next_v_preds), self.squeeze(rewards), self.squeeze(gaes_final)]

    # for ele in data :
    #   print(np.asarray(ele).shape)

    # v_preds = [ [cell[-1] for cell in row] for row in log_probs_And_v_preds]
    # next_v_preds = v_preds[1:] + [[0 for i in range(5)]]
    # gaes = self.get_gaes(rewards, v_preds, next_v_preds)
    # gaes = np.array(gaes).astype(dtype=np.float32)
    # gaes = (gaes - gaes.mean()) / gaes.std()


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
    output, value =  self.model(states)
    dist = self.get_dist(output)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    return log_probs, entropy, value

  def sample(self, data):
    sample_indices = np.random.randint(low=0, high=len(data[0]), size=self.batch_size)
    sampled_data = [np.take(a=a, indices=sample_indices, axis=0).astype(np.float32) for a in data]
    return sampled_data
    

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

    train_variables = self.model.trainable_variables
    grad = tape.gradient(total_loss, train_variables)  # compute gradient
    self.model_optimizer.apply_gradients(zip(grad, train_variables))
    print("total_loss = {:f} \t loss_clip = {:f} \t vf_loss = {:f} \t entropy = {:f}".format(total_loss, loss_clip, vf_loss, entropy) )
    


def check_nan(data):
  for ele in data:
    if np.any(np.isnan(ele)) : 
      return True
  return False


if __name__ == "__main__":
  mappo = MAPPO(n_updates = 128, batch_size=4)
  episodes = []


  while True:
    print("------------------- load model -----------------------")
    mappo.model = load_model("model")

    for _ in range(20):
      episodes.append({
        b"state": [np.random.rand(5, 1509)] * 100,
        b"action": [[random.randint(0, 17) for _ in range(5)]] * 100,
        b"reward": [[random.random() for _ in range(5)]] * 100,
        b"done": [[random.random() for _ in range(5)]] * 100,
        b"probs": [ np.random.rand(5, 18)] * 100,
        b"values": [[random.random() for _ in range(5)]] * 100,
      })  

    data = mappo.get_data(episodes)

    for i in range(mappo.n_updates):
      while True:
        sample = mappo.sample(data)
        if check_nan(sample) == False:
          break
      mappo.learn(*sample)

    print("------------------- save model -----------------------")
    mappo.save_model("model")

    # f = open("write.txt", "w")
    # print(mappo.model.trainable_variables, file=f)

