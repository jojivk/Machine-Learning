import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Define some hyper-parameters
learning_rate = 0.01
epochs =1000

display_step = 100

# First generate some random (X, Y) points, in the plane
N = 100
xrange = 100
yrange = 10
db = 0.0

train_X1 = xrange * np.random.random_sample(N)
train_X2 = xrange * np.random.random_sample(N)
#Here is the line Y = W1X1+ W2X2 + B. The computed value should be somehwere close
train_Y = 0.54 * train_X1 + 0.77* train_X2 + 2.5

#Define place holer for data that will be fed to the odel
X1= tf.placeholder("float")
X2= tf.placeholder("float")
Y= tf.placeholder("float")

#Define the vaiables that will be optimized by training
#Initialize them randomly
W1= tf.Variable(np.random.randn(), name="Weights1")
W2= tf.Variable(np.random.randn(), name="Weights2")
b =tf.Variable(np.random.randn(), name="Bias")

# Define the Graph, cost and optmizer
pred = tf.add(tf.add(tf.multiply(X2, W2), tf.multiply(X1, W1)), b)
cost = tf.reduce_mean(tf.pow(pred-Y, 2))/N
opti = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer();

# Now train the network
with tf.Session() as sess :
  sess.run(init)
  for ep in range(epochs):
    for (x1, x2, y) in zip(train_X1, train_X2, train_Y) :
      sess.run(opti, feed_dict={X1:x1, X2:x2, Y:y})
    if ((ep+1) % display_step == 0) :
      cst = sess.run(cost, feed_dict={X1:train_X1, X2:train_X2, Y:train_Y})
      print("Epoch :", ep+1," - Cost :", cst)
  print("Training Complete.....")
  cost = sess.run(cost, feed_dict={X1:train_X1, X2:train_X2, Y:train_Y})
  print("Cost  :", cost)
  print("Computed W1: ", sess.run(W1))
  print("Computed W2: ", sess.run(W2))
  print("Computed B: ", sess.run(b))
