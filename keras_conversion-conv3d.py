import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import time

np.random.seed(33)

dataset_dir = "./preprocessed_data/3D_CNN/raw_data/"

with open(dataset_dir+"1_108_shuffle_dataset_3D_win_10.pkl", "rb") as fp:
  	datasets = pickle.load(fp)
with open(dataset_dir+"1_108_shuffle_labels_3D_win_10.pkl", "rb") as fp:
  	labels = pickle.load(fp)
print("datasets shape:", datasets.shape)
datasets = datasets.reshape(len(datasets), 10, 10, 11, 1)
print("datasets shape after reshape: ", datasets.shape)
print("labels: ", np.unique(labels))
one_hot_labels = np.array(list(pd.get_dummies(labels)))
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)

print("labels after one hot encoding:", labels[1])

split = np.random.rand(len(datasets)) < 0.75

train_x = datasets[split]
train_y = labels[split]

train_sample = len(train_x)

test_x = datasets[~split] 
test_y = labels[~split]

test_sample = len(test_x)
print(train_x.shape)
print(test_x.shape)
print("**********("+time.asctime(time.localtime(time.time()))+") Load and Split dataset End **********\n")

print("**********("+time.asctime(time.localtime(time.time()))+") Define parameters and functions Begin: **********\n")

dropout_prob = 0.5
calibration = 'N'

n_labels = 5

training_epochs = 10

batch_size = 300
learning_rate = 1e-4

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv2D, Conv3D
from tensorflow.keras.layers import Reshape, Flatten, Softmax
from tensorflow.keras.optimizers import Adam

def model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    X_input = Input(shape = input_shape)
    
    conv_1 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding="same", strides=(1, 1, 1), activation="elu")(X_input)
    conv_2 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding="same", strides=(1, 1, 1), activation="elu")(conv_1)
    conv_3 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding="same", strides=(1, 1, 1), activation="elu")(conv_2)
    shape = conv_3.get_shape().as_list()
    
    pool_2_flat = Reshape([shape[1], shape[2]*shape[3]*shape[4]])(conv_3)
    fc = Dense(1024, activation="elu")(pool_2_flat)
    fc_drop = Dropout(dropout_prob)(fc)
    
    lstm_in = Reshape([10, 1024])(fc_drop)
    lstm_1 = LSTM(units=1024, return_sequences=True, unit_forget_bias=True, dropout=dropout_prob)(lstm_in)
    rnn_output = LSTM(units=1024, return_sequences=False, unit_forget_bias=True)(lstm_1)
    
    shape_rnn_out = rnn_output.get_shape().as_list()
    fc_out = Dense(shape_rnn_out[1], activation="elu")(rnn_output)
    fc_drop = Dropout(dropout_prob)(fc_out)
    y_ = Dense(n_labels)(fc_drop)
    y_posi = Softmax()(y_)

    model = Model(inputs = X_input, outputs = y_posi)
    return model

model = model(input_shape = (datasets.shape[1], datasets.shape[2], datasets.shape[3], datasets.shape[4]))

opt = Adam(lr=learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_path = "training_keras/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

print("**********("+time.asctime(time.localtime(time.time()))+") Model Fit Start **********\n")
history = model.fit(train_x, train_y, batch_size=300, epochs=300, shuffle=True, validation_data=(test_x, test_y), callbacks=[cp_callback])
print("**********("+time.asctime(time.localtime(time.time()))+") Model Fit End **********\n")

with open("./training_keras_hist/train_hist", "wb") as file:
    pickle.dump(history.history, file)