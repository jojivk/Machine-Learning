import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot


#Define some hyper-parameters
learning_rate = 0.01
epochs =1000

display_step = 100

# First generate some random (X, Y) points, in the plane
N = 100
xrange = 100
yrange = 10
db = 0.0

train_X = xrange * np.random.random_sample(N)
#Here is the line Y = WX+ B. The computed value should be somehwere close
train_Y = 0.54 * train_X + 2.5

#Define place holder for data that will be fed to the odel
X= tf.placeholder("float")
Y= tf.placeholder("float")

#Define the variables that will be optimized by training
#Initialize them randomly
W= tf.Variable(np.random.randn(), name="Weights")
b =tf.Variable(np.random.randn(), name="Bias")

# Define the Graph, cost and optmizer used
pred = tf.add(tf.multiply(X, W), b)
cost = tf.reduce_mean(tf.pow(pred-Y, 2))/N
opti = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer();

# Now train the network
with tf.Session() as sess :
  sess.run(init)
  for ep in range(epochs):
    for (x, y) in zip(train_X, train_Y) :
      sess.run(opti, feed_dict={X:x, Y:y})
    if ((ep+1) % display_step == 0) :
      cst = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
      print("Epoch :", ep+1," - Cost :", cst)
  print("Training Complete.....")
  cost = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
  print("Cost  :", cost)
  print("Computed W: ", sess.run(W))
  print("Computed B: ", sess.run(b))

  plot.plot(train_X, train_Y, 'ro', label='Data')
  plot.plot(train_X, sess.run(W) *train_X + sess.run(b), label='Fitted line')
  plot.legend()
  plot.show()
  
