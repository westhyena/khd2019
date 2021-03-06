import os
import argparse
import sys
import time
import random
import cv2
import numpy as np
import keras

from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from keras.layers import BatchNormalization, ReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import  ModelCheckpoint, ReduceLROnPlateau
# from keras.utils.training_utils import multi_gpu_model
import keras.backend.tensorflow_backend as K
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM


def inception_v3(in_shape, num_classes=4, dense_blocks=[64]):
  base_model = InceptionV3(include_top=False,
                             weights=None,
                             input_shape=in_shape,
                             classes=num_classes)

  x = base_model.output
  x = Flatten()(x)

  for node in dense_blocks:
    x = Dense(node, activation="relu")(x)
    x = Dropout(0.5)(x)

  if num_classes == 1:
    x = Dense(1, activation='sigmoid')(x)
  else:
    x = Dense(num_classes, activation='softmax')(x)
  
  model = Model(inputs=base_model.input, outputs=x)
  return model


def cnn_sample(in_shape, num_classes=4):    # Example CNN

    ## 샘플 모델
    # 입력 데이터가 (390, 307, 3)의 크기일 때의 예시 코드

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=in_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(ZeroPadding2D(padding=((0, 0), (0, 1))))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=5, padding='same'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=6, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=5, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add((ReLU()))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model
    
