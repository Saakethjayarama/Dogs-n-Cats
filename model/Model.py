import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Flatten, BatchNormalization, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import itertools
import os
import shutil
import random
import glob

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def trainModel():
    
  train_path = './datum/train'
  test_path = './datum/test'
  valid_path = './datum/valid'

  train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=['cats', 'dogs'], batch_size=10)
  test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=['cats', 'dogs'], batch_size=10, shuffle=False)
  valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cats', 'dogs'], batch_size=10, shuffle=False)

  imgs, labels = next(train_batches)

  model = Sequential([
      Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(224,224,3)), # Accepts image data, 3rd parama in input_shape specifies that there are 3 color channels
      MaxPool2D(pool_size=(2,2) , strides=2), # cuts down image dimensions into halves.
      Conv2D(filters=64, kernel_size=(3,3), activation='relu',  padding='same'),
      MaxPool2D(pool_size=(2, 2), strides=2),
      Flatten(), # Flatten everything to 1d tensor (1d uniform array)
      Dense(units=2, activation='softmax')
  ])
  model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

  model.save('./model/dogsncats.h5')

def predict(url):
  model = load_model('./model/dogsncats.h5')
  img = image.load_img(url, target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  result = np.argmax(model.predict(img))
  if result == 0:
    return "Cat"
  else:
    return "Dog"