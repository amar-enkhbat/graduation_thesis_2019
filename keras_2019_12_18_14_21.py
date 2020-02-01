# coding: utf-8
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sklearn
import os
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import time
random_state = 33
np.random.seed(random_state)

from datetime import datetime
now = datetime.now()
now = now.strftime("%Y_%m_%d_%H_%M")

results_path = "./results/keras_" + now
print("Saving results in:", results_path)
print()
import os
try:
    os.mkdir(results_path)
except OSError:
    print("Directory %s already exists. Creating new directory under %s(2)" % (results_path, results_path))
    os.mkdir(results_path+ "(2)")

# # Training data 64 channel 1-81

dataset_dir = "./dataset/preprocessed_dataset/"
with open(dataset_dir+"1_81_shuffle_dataset_3D_win_10.pkl", "rb") as fp:
    X_train = pickle.load(fp)
with open(dataset_dir+"1_81_shuffle_labels_3D_win_10.pkl", "rb") as fp:
    y_train = pickle.load(fp)
X_train = X_train.reshape(-1, 10, 10, 11, 1)
print("Dataset shape:", X_train.shape)
print("Labels shape:", y_train.shape)

print(X_train[0, 2].reshape(10, 11))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)

y_train = y_train.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)

# # Validation data 4-channel 82-108 

dataset_dir = "./dataset/preprocessed_dataset/"
result_dir = "./results/"
with open(dataset_dir+"82_108_shuffle_dataset_3D_win_10.pkl", "rb") as fp:
    X_valid = pickle.load(fp)
with open(dataset_dir+"82_108_shuffle_labels_3D_win_10.pkl", "rb") as fp:
    y_valid = pickle.load(fp)
X_valid = X_valid.reshape(-1, 10, 10, 11, 1)
print("Dataset shape:", X_valid.shape)
print("Labels shape:", y_valid.shape)

print(X_valid[0, 2].reshape(10, 11))

y_valid = y_valid.reshape(-1, 1)
y_valid = ohe.transform(y_valid)

with open(results_path + "/ohe", "wb") as file:
    pickle.dump(ohe, file)

# # Model

dropout_prob = 0.5
n_labels = y_train.shape[1]
batch_size = 300
learning_rate = 1e-4

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Conv2D, Conv3D, GRU
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
    
    conv_1 = Conv3D(filters=32, kernel_size=(1, 1, 1), padding="same", strides=(1, 1, 1), activation="elu")(X_input)
    conv_2 = Conv3D(filters=64, kernel_size=(1, 1, 1), padding="same", strides=(1, 1, 1), activation="elu")(conv_1)
    conv_3 = Conv3D(filters=128, kernel_size=(1, 1, 1), padding="same", strides=(1, 1, 1), activation="elu")(conv_2)
    shape = conv_3.get_shape().as_list()
    
    pool_2_flat = Reshape([shape[1], shape[2]*shape[3]*shape[4]])(conv_3)
    fc = Dense(1024, activation="elu")(pool_2_flat)
    fc_drop = Dropout(dropout_prob)(fc)
    
    lstm_in = Reshape([10, 1024])(fc_drop)
    # lstm_1 = LSTM(units=1024, return_sequences=True, unit_forget_bias=True, dropout=dropout_prob)(lstm_in)
    # rnn_output = LSTM(units=1024, return_sequences=False, unit_forget_bias=True)(lstm_1)
    lstm_1 = GRU(units=1024, return_sequences=True, dropout=dropout_prob)(lstm_in)
    rnn_output = GRU(units=1024, return_sequences=False)(lstm_1)
    
    shape_rnn_out = rnn_output.get_shape().as_list()
    fc_out = Dense(shape_rnn_out[1], activation="elu")(rnn_output)
    fc_drop = Dropout(dropout_prob)(fc_out)
    y_ = Dense(n_labels)(fc_drop)
    y_posi = Softmax()(y_)
    
    model = Model(inputs = X_input, outputs = y_posi)
    return model

model = model(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))
opt = Adam(lr=learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print(model.summary())

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_path = results_path + "/model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

training_start_time = datetime.now()
print("Training start date and time:", training_start_time)
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=300, shuffle=True, validation_data=(X_valid, y_valid), callbacks=[cp_callback])
training_end_time = datetime.now()
print("Training end date and time:", training_end_time)
model.save(results_path + "/model/model.h5")

with open(results_path + "/train_hist", "wb") as file:
    pickle.dump(history.history, file)

print("Training start date and time:", training_start_time)
print("Training end date and time:", training_end_time)
print("Training duration:", training_end_time - training_start_time)
