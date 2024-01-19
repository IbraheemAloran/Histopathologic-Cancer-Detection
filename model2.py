import os
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
import fnmatch
import random
import visualkeras
import cv2 as cv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import RandomFlip, RandomZoom, RandomRotation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam


#Getting data
train_path = '../input/histopathologic-cancer-detection/train/'
test_path = '../input/histopathologic-cancer-detection/test/'

train_data = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')
sample_submission = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

test_data = pd.DataFrame({'id':os.listdir(test_path)})

train_data['id'] = train_data['id'] + '.tif'
train_data['label'] = train_data['label'].astype(str)

datagen = ImageDataGenerator(rescale=1./255.,
                            validation_split=0.15)

batchSize = 256


# generate training data
train_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=train_path,
    x_col="id",
    y_col="label",
    subset="training",
    batch_size=batchSize,
    seed=985723,
    class_mode="binary",
    target_size=(64,64))  



# generate validation data
valid_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=train_path,
    x_col="id",
    y_col="label",
    subset="validation",
    batch_size=batchSize,
    seed=985723,
    class_mode="binary",
    target_size=(64,64))  



model2_auc = tf.keras.metrics.AUC()
    
model2 = Sequential()

model2.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model2.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.1))

model2.add(BatchNormalization())
model2.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model2.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.1))

model2.add(BatchNormalization())
model2.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model2.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(BatchNormalization())
model2.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model2.add(Flatten())
model2.add(Dense(1, activation='sigmoid'))

#build model by input size
model2.build(input_shape=(batchSize, 64, 64, 3))

#compile
adam_optimizer = Adam(learning_rate=0.0001)
model2.compile(loss='binary_crossentropy', metrics=['accuracy', model2_auc], optimizer=adam_optimizer)

# Summary of the 2nd model
model2.summary()

visualkeras.layered_view(model2, type_ignore=[ZeroPadding2D, Flatten], legend=True)

EPOCHS = 10

# train model
history_model2 = model2.fit(
                        train_generator,
                        epochs = EPOCHS,
                        validation_data = valid_generator)

