import time
import random
import datetime
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Dropout, MaxPool2D,
    Flatten, Dense, Input, Concatenate, LeakyReLU, Add, Activation, Conv2DTranspose
)

K.tensorflow_backend._get_available_gpus()

train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/submission.csv')

X_train = (train[[str(i) for i in range(784)]] / 255.).values.reshape(-1, 28, 28, 1)
y_train = to_categorical(train['digit'].values)


datagen = ImageDataGenerator(
        zca_epsilon=1e-06, 
        rotation_range=10,  
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1, 
        zoom_range=0.1)


# public : 92.156

model = Sequential()
model.add(Conv2D(64, (3, 3), padding="same", input_shape=(28,28,1)))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), strides=(2,2)))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(256,(3, 3), strides=(2,2)))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(512,(3, 3), padding="same"))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(512,(3, 3)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(0.02))
model.add(Dense(512))
model.add(Activation('relu'))

model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.02))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x) 

epochs = 100

X_train2, X_val2, y_train2, y_val2 = train_test_split(
    X_train, y_train, test_size = 0.1) 

history = model.fit_generator(
    datagen.flow(X_train2, y_train2, batch_size=32),
    epochs=epochs, 
    steps_per_epoch=X_train2.shape[0]//32,
    validation_data=(X_val2, y_val2), 
    callbacks=[annealer], 
    verbose=1
)

''' Train accuracy=0.95394, Validation accuracy=0.90244
 
model = Sequential()
model.add(Conv2D(64, (3, 3), padding="same", input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), strides=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(256,(3, 3), strides=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(512,(3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(512,(3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(BatchNormalization(axis=-1))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


annealer = LearningRateScheduler(lambda x: 1e-3 * 0.98 ** x) 

epochs = 100

X_train2, X_val2, y_train2, y_val2 = train_test_split(
    X_train, y_train, test_size = 0.1) 

history = model.fit_generator(
    datagen.flow(X_train2, y_train2, batch_size=128),
    epochs=epochs, 
    steps_per_epoch=X_train2.shape[0]//128,
    validation_data=(X_val2, y_val2), 
    callbacks=[annealer], 
    verbose=1
)


'''




print(
    f"CNN: Epochs={epochs:d}, " +
    f"Train accuracy={max(history.history['accuracy']):.5f}, " +
    f"Validation accuracy={max(history.history['val_accuracy']):.5f}"
)


model.save_weights(f'params.h5')
    
model_json = model.to_json()
with open(f"model.json", "w") as json_file : 
    json_file.write(model_json)

X_test = (test[[str(i) for i in range(784)]] / 255.).values.reshape(-1, 28, 28, 1)
results = model.predict(X_test)
print(results)
print(results.argmax(axis=1))
result = results.argmax(axis=1)
submission.digit = result
submission.to_csv('result.csv', index=False) 
