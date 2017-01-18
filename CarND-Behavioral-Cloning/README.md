### Preprocessing & Augmentation

Dataset provided by Udacity has set of 3 images and along with steering angle.

#### Augmentation Mechanism

   * Center Image
      * Convert the image to HSV of size (200, 66)
      * Flip the image and the steering angle
   * Left & Right Image
      * Adjust the left image by adding 0.3 (correction angle) and follow the same steps for center image
      * Adjust the right image by subtracting 0.3(correction angle) and follow the same steps for center image
   * Normalize the image to have values between 0 and 1.   


### Models Exploration

After reading through multiple blogs, slack and Udacity forums. I started with the nvidia model but apparently other members got to drive on both the tracks using a smaller network.

Modified Nvidia network:

```
  image_model = Sequential()
  image_model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', input_shape=input_shape))
  image_model.add(ELU())
  image_model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid', input_shape=input_shape))
  image_model.add(ELU())
  image_model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', input_shape=input_shape))
  image_model.add(ELU())
  image_model.add(Convolution2D(96, 3, 3, subsample=(1,1), border_mode='valid', input_shape=input_shape))
  image_model.add(ELU())
  image_model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
  image_model.add(Dropout(0.25))
  image_model.add(Flatten())
  image_model.add(Dense(1024))
  image_model.add(Dense(100))
  image_model.add(Dense(50))
  image_model.add(Dense(1))
  image_model.compile(optimizer='adam', loss='mse')
```

Simpler Network - V1
Reduced the image size to (12, 32, 3) 

```
  image_model = Sequential()
  image_model.add(Convolution2D(64, 3, 3, subsample=(2,2), border_mode='valid', input_shape=input_shape))
  image_model.add(ELU())
  image_model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
  image_model.add(Dropout(0.25))
  image_model.add(Flatten())
  image_model.add(Dense(128))
  image_model.add(Dense(1))
  image_model.compile(optimizer='adam', loss='mse')
```


Simple Network - v2

```
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
```

### Model Architecture: 

Used Adam optimizer default configuration.
Batch Size: 128
Image shape : (12, 32, 3)
Train on 43394 samples, validate on 4822 samples

It's interesting to see how we can reduce the size of the image from (160,320,3) to (12, 32, 3) and still preserve all the necessary information to build 
a working network.

Add MaxPooling to reduce the dimension and Dropout to prevent overfitting.

EDA notebook has some more exploration on the steering angle and the results of the preprocessed images.
 
```
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 5, 15, 64)     1792        convolution2d_input_1[0][0]
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 5, 15, 64)     0           convolution2d_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 4, 14, 64)     0           elu_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 4, 14, 64)     0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 3584)          0           dropout_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           917760      flatten_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 256)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 128)           32896       dropout_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             129         dense_2[0][0]
====================================================================================================
Total params: 952577

```

Results

[Video](https://www.youtube.com/watch?v=hRSRT1hJtpI)
