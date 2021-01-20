from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
from random import Random,shuffle
from tqdm import tqdm

from keras import backend as K
import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

'''Setting up the env'''

### Edit Disini
IMG_SIZE = 64       # Ukuran gambar (u/ diperkecil)
LR = 0.25           # Learning Rate
no_batch = 1        # Banyaknya gambar u/ sekali proses
no_epoch = 30       # Banyaknya epoch
LABELS = {"100celeng", "25s75c", "50s50c", "75s25c", "100sapi"} # Nama-nama kelas
#################
'''Setting up the model which will help with tensorflow models'''
MODEL_NAME = 'dagingtrain-{}-e{}-img{}.model'.format(LR, no_epoch, IMG_SIZE)   # Nama model

'''Loading data'''
train_data, train_labels = np.load('train_data.npy'), np.load('new_labels.npy')
test_data, test_labels = np.load('test_data.npy'), np.load('test_labels.npy')
# train_data, train_labels = np.load('train_data_128.npy'), np.load('new_labels_128.npy')
# test_data, test_labels = np.load('test_data_128.npy'), np.load('test_labels_128.npy')

'''Creating the neural network using tensorflow'''
import keras
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D

## Kode CNN
classifier = Sequential()
# Conv2D('jumlah kernel', '(size kernel)', 'bentuk input', 'fungsi aktifasi')
classifier.add(Conv2D(32, (3,3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())  # Proses Flattening
classifier.add(Dense(units = 128, activation = 'relu'))  # Jumlah node pada hidden layer
classifier.add(Dense(units = 5, activation = 'sigmoid')) # Jumlah node output (Jumlah kelas data)

trainX, testX, trainY, testY = train_data, test_data, train_labels, test_labels
# ### construct the training image generator for data augmentation
aug = ImageDataGenerator(rescale = 1.)
train_aug = aug.flow(trainX, trainY, batch_size=no_batch)
aug = ImageDataGenerator(rescale = 1.)
test_aug = aug.flow(testX, testY, batch_size=no_batch)

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print("[INFO] compiling model..."+str(len(trainX)))
classifier.fit_generator(train_aug, steps_per_epoch = len(trainX)//no_batch, epochs = no_epoch, validation_data = test_aug, validation_steps=len(testX))

classifier.save(MODEL_NAME)             # Proses untuk men-save model
############
# K.set_learning_phase(0)
# classifier = load_model(MODEL_NAME)   # Proses untuk me-load model
############

## Making Predictions (PROSES UNTUK AKURASI DATA TESTING) ##
# predictions = classifier.predict(testX, batch_size=no_batch)
# # # print(predictions)      # Hasil kelas dari Prediksi
# # # print(testY)            # Kelas aslinya (Data test)
# print(confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1)))
# # # ^ Kesimpulan hasil dari prediksi
# #
# print("Accuracy score: ")
# print(accuracy_score(testY.argmax(axis=1),predictions.argmax(axis=1)))
# print("\nTotal test data: "+str(len(testX)))
