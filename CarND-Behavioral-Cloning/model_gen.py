import pickle
import tensorflow as tf
import json
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten,ELU
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers import Input, Flatten, Dense
from keras.models import Model, Sequential
from keras.models import model_from_json
import pandas as pd
from keras.callbacks import ModelCheckpoint
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
import matplotlib.pyplot as plt
import matplotlib.image as mpg
# command line flags
flags.DEFINE_string('image_dir', '', "Image directory path")
flags.DEFINE_string('log', '', "Driving Log")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def preprocess(image):
  img_shape = image.shape
  live = False
  if len(image.shape) == 4:
    live = True
    image = image.reshape(image.shape[1:])

  height = image.shape[0]
  width = image.shape[1]
  cropped = image[50:140,:,:]
  resized = cv2.resize(cropped, (32, 12))
  #resized = cv2.resize(cropped, (int(width/4), int((height-40)/4)))
  resized = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
  # resized = tf.image.resize_images(cropped, (int(width/2), int((height-40)/2)))
  if live:
    resized = resized.reshape(img_shape[:1] + resized.shape)

  return resized/255.0 - 0.5


def flipped(image, steering_angle):
    return (cv2.flip(image, 1), -1*steering_angle)

def data_generator(X_data, image_dir, batch_size):
  while True:
    x = []
    y = []
    curr_batch_size = 0
    for i in range(batch_size):
      if (i + curr_batch_size) > len(X_data):
        X_data.iloc[np.random.permutation(len(X_data))]
        curr_batch_size = 0

      x_pre, y_pre = load_images(X_data.iloc[i], image_dir)
      x += x_pre
      y += y_pre
      curr_batch_size += batch_size

    yield np.array(x), np.array(y)

def load_images(row, image_dir):

  steering_angle = row['steering']
  ANGLE_CORRECTION = 0.3
  x = []
  y = []
  # Read center image
  filename = image_dir+"/"+row['center'].strip()
  x_pre, y_pre = load_single_image(filename, steering_angle, 0)
  x += x_pre
  y += y_pre

  # Read Left Camera image
  filename = image_dir+"/"+row['left'].strip()
  x_pre, y_pre = load_single_image(filename, steering_angle, ANGLE_CORRECTION)
  x += x_pre
  y += y_pre

  # Read right camera image
  filename = image_dir+"/"+row['right'].strip()
  x_pre, y_pre = load_single_image(filename, steering_angle, -ANGLE_CORRECTION)
  x += x_pre
  y += y_pre

  return x,y

def load_single_image(filename, steering_angle, adjust_angle):
  x = []
  y = []
  img_data = preprocess(plt.imread(filename))
  x.append(img_data)
  y.append(steering_angle)

  # Flipped the imag and reverse angle
  fdata = flipped(img_data, steering_angle)
  x.append(fdata[0])
  y.append(fdata[1])
  return x,y

def load_data(df, image_dir):
    y_true = df[['steering']]
    img_names = df[['center']]
    x = []
    y = []
    count = 0
    ANGLE_CORRECTION = 0.3
    for row in df.itertuples():
        filename = image_dir+"/"+row[1]
        img_data = preprocess(plt.imread(filename))
        x.append(img_data)
        y.append(y_true.iloc[count])
        fdata = flipped(img_data, y_true.iloc[count])
        x.append(fdata[0])
        y.append(fdata[1])
        count += 1

    count = 0
    img_names = df[['left']]
    for row in img_names.itertuples():
      filename = image_dir+"/"+str.strip(row[1])
      img_data = preprocess(plt.imread(filename))
      x.append(img_data)
      y.append(y_true.iloc[count] + ANGLE_CORRECTION)
      fdata = flipped(img_data, y_true.iloc[count] + ANGLE_CORRECTION)
      x.append(fdata[0])
      y.append(fdata[1])
      count += 1

    count = 0
    img_names = df[['right']]
    for row in img_names.itertuples():
      filename = image_dir+"/"+str.strip(row[1])
      img_data = preprocess(plt.imread(filename))
      x.append(img_data)
      y.append(y_true.iloc[count] - ANGLE_CORRECTION)
      fdata = flipped(img_data, y_true.iloc[count] - ANGLE_CORRECTION)
      x.append(fdata[0])
      y.append(fdata[1])
      count += 1

    return np.array(x), np.array(y)

def train_model(input_shape):
  print(input_shape)
  image_model = Sequential()
  image_model.add(Convolution2D(64, 3, 3, subsample=(2,2), border_mode='valid', input_shape=input_shape))
  image_model.add(ELU())
  image_model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
  image_model.add(Dropout(0.25))
  image_model.add(Flatten())
  image_model.add(Dense(128))
  image_model.add(Dense(1))
  image_model.compile(optimizer='adam', loss='mse')
  return image_model

def main(_):
  print("Image directory :", FLAGS.image_dir)
  print("log :", FLAGS.log)
  print("Epochs :", FLAGS.epochs)
  print("Batch Size :", FLAGS.batch_size)
  df = pd.read_csv(FLAGS.log)

  # if os.path.isfile('x_preprocessed.data.npy') and os.path.isfile('y_preprocessed.data.npy') :
  #   print("****** Preprocessed file already exists. Loading X_train and y_train ******")
  #   X_train = np.load("x_preprocessed.data.npy")
  #   Y_train = np.load("y_preprocessed.data.npy")
  # else:
  #   print("****** Preprocessed data doesn't exist!!!. Starting preprocessing ******")
  #   X_train, Y_train = load_data(FLAGS.image_dir, FLAGS.log)
  #   print("****** Saving preprocessed data ******")
  #   np.save("x_preprocessed.data", X_train)
  #   np.save("y_preprocessed.data", Y_train)

  # print(X_train.shape)
  # print(Y_train.shape)
  # input_shape = X_train.shape[1:4]
  np.random.seed(0)

  df = pd.read_csv(FLAGS.log)
  train, validation = train_test_split(df, test_size = 0.1)
  train_generator = data_generator(train, FLAGS.image_dir, FLAGS.batch_size)
  validation_generator = data_generator(validation, FLAGS.image_dir, FLAGS.batch_size)

  # for t in train_generator:
  #   print(t)
  input_shape = (12, 32, 3)
  model = train_model(input_shape)
  checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
  model.fit_generator(train_generator, samples_per_epoch=19200, nb_epoch = 10, validation_data=validation_generator, nb_val_samples=2048)
  #model.fit(X_train, Y_train, validation_split=0.1, shuffle=True, nb_epoch=FLAGS.epochs, callbacks=[checkpointer])
  model.summary()
  with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
  model.save_weights('model.h5')
if __name__ == '__main__':
    tf.app.run()
