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
  resized = cv2.resize(cropped, (int(width/10), int((height-40)/10)))
  resized = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
  # resized = tf.image.resize_images(cropped, (int(width/2), int((height-40)/2)))
  if live:
    resized = resized.reshape(img_shape[:1] + resized.shape)

  return resized/255.0 - 0.5


def flipped(image, steering_angle):
    return (cv2.flip(image, 1), -1*steering_angle)

def train_generator(X_data, image_dir, batch_size):
  while True:
    x = []
    y = []
    curr_batch_size = 0
    for i in range(batch_size):
      img_path = X_data.iloc[i]['center']
      angle = X_data.iloc[i]['steering']
      filename = image_dir+"/"+img_path
      img_data = preprocess(plt.imread(filename))
      x.append(img_data)
      y.append(angle)

    yield np.array(x), np.array(y)

def load_data(image_dir, log):
    df = pd.read_csv(log)
    y_true = df[['steering']]
    img_names = df[['center']]
    x = []
    y = []
    count = 0
    ANGLE_CORRECTION = 0.30
    for row in img_names.itertuples():
        filename = image_dir+"/"+row[1]
        img_data = preprocess(plt.imread(filename))
        x.append(img_data)
        y.append(y_true.iloc[count])
        fdata = flipped(img_data, y_true.iloc[count])
        x.append(fdata[0])
        y.append(fdata[1])
        count += 1
        # if count > 10:
        #   break
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
      # if count > 10:
      #     break

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
      # if count > 10:
      #     break

    return np.array(x), np.array(y)

def train_model(input_shape):
  print(input_shape)
  image_model = Sequential()
  image_model.add(Convolution2D(64, 3, 3, subsample=(2,2), border_mode='valid', input_shape=input_shape))
  image_model.add(ELU())
  image_model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
  image_model.add(Dropout(0.25))
  image_model.add(Flatten())
  image_model.add(Dense(256))
  image_model.add(Dropout(0.25))
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

  if os.path.isfile('x_preprocessed.data.npy') and os.path.isfile('y_preprocessed.data.npy') :
    print("****** Preprocessed file already exists. Loading X_train and y_train ******")
    X_train = np.load("x_preprocessed.data.npy")
    Y_train = np.load("y_preprocessed.data.npy")
  else:
    print("****** Preprocessed data doesn't exist!!!. Starting preprocessing ******")
    X_train, Y_train = load_data(FLAGS.image_dir, FLAGS.log)
    print("****** Saving preprocessed data ******")
    np.save("x_preprocessed.data", X_train)
    np.save("y_preprocessed.data", Y_train)

  print(X_train.shape)
  print(Y_train.shape)
  input_shape = X_train.shape[1:4]

  np.random.seed(0)
  model = train_model(input_shape)
  checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
  model.fit(X_train, Y_train, validation_split=0.1, shuffle=True, nb_epoch=FLAGS.epochs, callbacks=[checkpointer])
  model.summary()
  with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
  model.save_weights('model.h5')
if __name__ == '__main__':
    tf.app.run()
