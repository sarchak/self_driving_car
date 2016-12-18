import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

def one_hot(labels):
    n_values = np.max(labels) + 1
    return np.eye(n_values)[labels]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def shuffle_data(X, y):
  train_idx = np.arange(X.shape[0])
  np.random.shuffle(train_idx)
  X = X[train_idx]
  y = y[train_idx]
  return (X, y)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def conv_net(x, weights, biases, keep_prob):
    W_conv1 = weight_variable([3, 3, 1, 64])
    b_conv1 = bias_variable([64])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3, 3, 64, 128])
    b_conv2 = bias_variable([128])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([3, 3, 128, 256])
    b_conv3 = bias_variable([256])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #W_fc1 = weight_variable([8 * 8 * 128, 1024])
    #b_fc1 = bias_variable([1024])

    W_fc1 = weight_variable([4 * 4 * 256, 1024])
    b_fc1 = bias_variable([1024])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 512])
    b_fc2 = bias_variable([512])

    fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    h_fc2_drop = tf.nn.dropout(fc2, keep_prob)

    W_fc3 = weight_variable([512, 43])
    b_fc3 = bias_variable([43])

    out = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    return out

training_file = './train.p'
testing_file = './test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
y_train = one_hot(y_train)
y_test = one_hot(y_test)
X_train = rgb2gray(X_train).reshape(-1, 32, 32, 1)
X_test = rgb2gray(X_test).reshape(-1, 32, 32, 1)

X_train, y_train = shuffle_data(X_train, y_train)
X_test, y_test = shuffle_data(X_test, y_test)

# Normalize
X_train = X_train/255
X_test = X_test/255

n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = y_train.shape[1]

batch_size = 64

x = tf.placeholder("float", [None, 32, 32, 1])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

weights = {}
biases = {}
num_epochs = 100
logits = conv_net(x, weights, biases, keep_prob)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("====== Starting Training")
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for t in range(num_epochs):
        for i in range(int(n_train/batch_size)):
          next_index = i * batch_size
          batch_x = X_train[next_index: next_index + batch_size]
          batch_y = y_train[next_index: next_index + batch_size]
          if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch_x, y: batch_y, keep_prob: 1})
            print("step %d, training accuracy %g"%(i, train_accuracy))
          train_step.run(feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: X_test, y: y_test, keep_prob: 1.0}))

