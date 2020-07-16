import random
import tensorflow as tf

inputs = [[random.random() for x in range(1509)] for _ in range(5)]

model = tf.saved_model.load("slmodel/20200716-160514")

f = open("read.txt", "w")
print(model.trainable_variables, file=f)