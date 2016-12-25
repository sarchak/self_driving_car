
from sklearn.utils import shuffle
import numpy as np
import pickle
import cv2
import csv
from datetime import datetime
from scipy.misc import imread
test_images = []

training_file = './train.p'
testing_file = './test.p'


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

X_train = rgb2gray(X_train).reshape(-1, 32, 32, 1)
X_test = rgb2gray(X_test).reshape(-1, 32, 32, 1)

X_train = X_train/255
X_test=X_test/255



for i in range(0,15):
    image = imread("test"+str(i)+".png")
    print("Reading image "+ "test"+str(i)+".png")
    image = image.astype(np.uint8)
    test_images.append(image)
test_data = np.array(test_images)
test_data = rgb2gray(test_data).reshape(-1,32,32,1)
test_data = test_data/255

print(X_test[0][0:1])
print(test_data[0][0:1])

#test_data = rgb2gray(test_data).reshape(-1, 32, 32, 1)
#test_data = test_data / 255
