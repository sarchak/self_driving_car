import pickle
import tensorflow as tf
import json
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten,ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Input, Flatten, Dense
from keras.models import Model, Sequential
from keras.models import model_from_json
import pandas as pd
from keras.callbacks import ModelCheckpoint
import cv2
from PIL import Image
import sys

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
  cropped = image[40:height,:,:]
  resized = cv2.resize(cropped, (int(width/2), int((height-40)/2)))
  # resized = tf.image.resize_images(cropped, (int(width/2), int((height-40)/2)))
  if live:
    resized = resized.reshape(img_shape[:1] + resized.shape)
    print(resized.shape)
  return resized/255.0 - 0.5

def flipped(image, steering_angle):
    print("Steering angle ", steering_angle)
    return (cv2.flip(image, 1), -1*steering_angle)

def load_data(image_dir, log):
    df = pd.read_csv(log)
    y_true = df[['steering']]
    img_names = df[['center']]
    x = []
    y = []
    count = 0
    for row in img_names.itertuples():
        filename = image_dir+"/"+row[1]
        img_data = preprocess(plt.imread(filename))
        x.append(img_data)
        #y.append(y_true.iloc[count])
        # fdata = flipped(img_data, y_true.iloc[count])
        # x.append(fdata[0])
        # y.append(fdata[1])
        count += 1
        # if count > 000:
        #   break

    return np.array(x), np.array(y_true[0:count])

def train_model(input_shape):
  print(input_shape)
  image_model = Sequential()
  image_model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=input_shape))
  image_model.add(ELU())
  image_model.add(MaxPooling2D(pool_size=(2,2)))
  image_model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=input_shape))
  image_model.add(ELU())
  image_model.add(MaxPooling2D(pool_size=(2,2)))
  image_model.add(Convolution2D(256, 3, 3, border_mode='same', input_shape=input_shape))
  image_model.add(ELU())
  image_model.add(MaxPooling2D(pool_size=(2,2)))
  image_model.add(Flatten())
  image_model.add(Dense(128))
  image_model.add(Dense(64))
  image_model.add(Dense(1))
  image_model.compile(optimizer='adam', loss='mse')
  return image_model

def main(_):
  print("Image directory :", FLAGS.image_dir)
  print("log :", FLAGS.log)
  print("Epochs :", FLAGS.epochs)
  print("Batch Size :", FLAGS.batch_size)
  X_train, Y_train = load_data(FLAGS.image_dir, FLAGS.log)
  print(X_train.shape)
  print(Y_train.shape)
  input_shape = X_train.shape[1:4]
  model = train_model(input_shape)
  checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
  model.fit(X_train, Y_train, validation_split=0.15, shuffle=True, nb_epoch=FLAGS.epochs, callbacks=[checkpointer])

  with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
  model.save_weights('model.h5')
if __name__ == '__main__':
    tf.app.run()
