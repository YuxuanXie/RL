import tensorflow as tf

tf.reset_default_graph()

model = tf.saved_model.load("sv8feature0")

# outputs = imported(inputs)

