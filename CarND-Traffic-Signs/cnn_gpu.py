import numpy as np
import tensorflow as tf
import pickle
import cv2
from datetime import datetime

save_file = 'model.ckpt'

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

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation
    t1 = datetime.now()

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    # Denoise
    #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    t2 = datetime.now()


    return img


def accuracy_func(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def augmented_data(X, y, samples=2):
  x_shape = X.shape
  y_shape = y.shape
  X_train_augmented = []
  y_train_augmented = []
  idx = 0
  print("************* Starting Augmenting data")
  for (img, label) in zip(X_train, y_train):
    idx += 1
    tmp_x = []
    tmp_y = []
    for i in range(0,samples):
        img = transform_image(img, 20,10,5)
        tmp_x.append(img)
        tmp_y.append(label)
    X_train_augmented.append(tmp_x)
    y_train_augmented.append(tmp_y)

  X_train_augmented = np.array(X_train_augmented,dtype = np.float32()).reshape([len(X_train)*samples, 32, 32, 3])
  y_train_augmented = np.array(y_train_augmented,dtype = np.int32()).reshape(len(y_train)*samples,)
  return(X_train_augmented, y_train_augmented)

weight_layers = {'layer1': 64, 'layer2':128, 'layer3': 256}
biases_layers = {'layer1': 64, 'layer2':128, 'layer3': 256}

def conv_net(x, weights, biases, keep_prob):
    W_conv1 = weight_variable([5, 5, 1, weight_layers['layer1']])
    b_conv1 = bias_variable([biases_layers['layer1']])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, weight_layers['layer1'], weight_layers['layer2']])
    b_conv2 = bias_variable([biases_layers['layer2']])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([5, 5, weight_layers['layer2'], weight_layers['layer3']])
    b_conv3 = bias_variable([biases_layers['layer3']])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #W_fc1 = weight_variable([8 * 8 * 128, 1024])
    #b_fc1 = bias_variable([1024])

    W_fc1 = weight_variable([4 * 4 * weight_layers['layer3'], 1024])
    b_fc1 = bias_variable([1024])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * weight_layers['layer3']])
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
X_train, y_train = augmented_data(X_train, y_train)
X_train = rgb2gray(X_train).reshape(-1, 32, 32, 1)
X_test = rgb2gray(X_test).reshape(-1, 32, 32, 1)

y_train = one_hot(y_train)
y_test = one_hot(y_test)

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

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

#train_step = tf.train.AdagradOptimizer(1e-2).minimize(cross_entropy)
train_prediction = tf.nn.softmax(logits)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

validation_size = int(0.05 * n_train)

X_train = X_train[0:-validation_size]
y_train = y_train[0:-validation_size]

X_validation = X_train[-validation_size:]
y_validation = y_train[-validation_size:]

print("***************** Data shape ********************")
print("X_train Size {}".format(X_train.shape))
print("Y Size {}".format(y_train.shape))
print("X_validation Size : {}".format(X_validation.shape))
print("Y validation_size : {}".format(y_validation.shape))
saver = tf.train.Saver()
print("***************** Starting Training *************")
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for t in range(num_epochs):
        for i in range(int(n_train/batch_size)):
          next_index = i * batch_size
          batch_x = X_train[next_index: next_index + batch_size]
          batch_y = y_train[next_index: next_index + batch_size]
          # _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
          # if (i % 100 == 0):
          #   test_accuracy = accuracy.eval(feed_dict={
          #                 x: X_test, y: y_test, keep_prob: 1.0})
          #   print('Minibatch loss at step %d: %f, Minibatch accuracy: %.1f' % (i, l, accuracy_func(predictions, batch_y)))
          #   print('Validation accuracy: %.1f' % test_accuracy)
          if i%100 == 0:
            valid_accuracy = accuracy.eval(feed_dict={
                x:X_validation, y: y_validation, keep_prob: 1})
            print("step %d, validation accuracy %g"%(i, valid_accuracy))
            print("test accuracy %g"%accuracy.eval(feed_dict={
              x: X_test, y: y_test, keep_prob: 1.0}))
          optimizer.run(feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
    print("***************** Test Accuracy *************")
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: X_test, y: y_test, keep_prob: 1.0}))
    saver.save(sess, save_file)
