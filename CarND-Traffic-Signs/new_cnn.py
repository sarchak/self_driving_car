import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import pickle
import cv2
from datetime import datetime
EPOCHS = 40
BATCH_SIZE = 64 
from tensorflow.contrib.layers import flatten

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

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

def augmented_data(X, y, samples=2):
  x_shape = X.shape
  y_shape = y.shape
  X_train_augmented = []
  y_train_augmented = []
  idx = 0
  print("************* Starting Augmenting data")
  for (img, label) in zip(X_train, y_train):
    idx += 1
    tmp_x = [img]
    tmp_y = [label]
    for i in range(0,samples):
        img = transform_image(img, 20,10,5)
        tmp_x.append(img)
        tmp_y.append(label)
    X_train_augmented.append(tmp_x)
    y_train_augmented.append(tmp_y)

  X_train_augmented = np.array(X_train_augmented,dtype = np.float32()).reshape([len(X_train)*(samples+1), 32, 32, 3])
  y_train_augmented = np.array(y_train_augmented,dtype = np.int32()).reshape(len(y_train)*(samples+1),)
  return(X_train_augmented, y_train_augmented)


training_file = './train.p'
testing_file = './test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
X_train, y_train = shuffle(X_train, y_train)

#X_train, y_train = augmented_data(X_train, y_train)

X_train = rgb2gray(X_train).reshape(-1, 32, 32, 1)
X_test = rgb2gray(X_test).reshape(-1, 32, 32, 1)

X_train = X_train/255
X_test=X_test/255
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape
print(y_train.shape)
# TODO: How many unique classes/labels there are in the dataset.
n_classes = y_train.shape[0]

validation_size = int(0.01 * n_train)

X_train = X_train[0:-validation_size]
y_train = y_train[0:-validation_size]


X_validation = X_train[-validation_size:]
y_validation = y_train[-validation_size:]

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


print("Updated Image Shape: {}".format(X_train[0].shape))

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

weight_layers = {'layer1': 64, 'layer2':128, 'layer3': 256}
biases_layers = {'layer1': 64, 'layer2':128, 'layer3': 256}

def conv_net(x, keep_prob):
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



x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)

one_hot_y = tf.one_hot(y, 43)
rate = 0.0001

logits = conv_net(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    saver.save(sess, 'cnn_model')
    print("Model saved")
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
