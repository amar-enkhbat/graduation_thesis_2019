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
try:
    os.mkdir(results_path)
except OSError:
    print("Directory %s already exists. Creating new directory under %s(2)" % (results_path, results_path))
    os.mkdir(results_path+ "(2)")

dataset_dir = "./dataset/preprocessed_dataset/"
result_dir = "./results/"
with open(dataset_dir+"1_108_1x4_dataset_3D_win_10_normalize_False_overlap_False.pkl", "rb") as fp:
    dataset = pickle.load(fp)
with open(dataset_dir+"1_108_1x4_label_3D_win_10_normalize_False_overlap_False.pkl", "rb") as fp:
    labels = pickle.load(fp)
height = dataset.shape[2]
width = dataset.shape[3]
window_size = dataset.shape[1]
print("Dataset shape:", dataset.shape)
print("Labels shape:", labels.shape)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(dataset, labels, test_size=0.25, random_state=random_state, shuffle=True)
print("Train dataset shape:", X_train.shape)
print("Train label shape:", y_train.shape)
print("Test dataset shape:", X_valid.shape)
print("Test label shape:", y_valid.shape)

print("Dataset example:")
print(X_train[0, 2].reshape(height, width))
print(X_valid[0, 2].reshape(height, width))

# Label encoding

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)

y_train = y_train.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_valid = y_valid.reshape(-1, 1)
y_valid = ohe.transform(y_valid)

with open(results_path + "/ohe", "wb") as file:
    pickle.dump(ohe, file)

# Dataset Normalization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

original_X_train_shape = X_train.shape
original_X_valid_shape = X_valid.shape

X_train = X_train.reshape(-1, height*width)
X_valid = X_valid.reshape(-1, height*width)
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_train = X_train.reshape(-1, window_size, height, width, 1)
X_valid = X_valid.reshape(-1, window_size, height, width, 1)

with open(results_path + "/scaler", "wb") as file:
    pickle.dump(scaler, file)
print("Dataset example after normalization:")
print(X_train[0, 2].reshape(height, width))
print(X_valid[0, 2].reshape(height, width))

# Model

dropout_prob = 0.5
n_labels = y_train.shape[1]
batch_size = 300
learning_rate = 1e-4

filters = 128
kernel_size = (1, 1, 1)
recurrent_units = 1024

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Conv2D, Conv3D, GRU
from tensorflow.keras.layers import Reshape, Flatten, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional

def model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    X_input = Input(shape = input_shape)
    
    # conv_1 = Conv3D(filters=32, kernel_size=(1, 1, 1), padding="same", strides=(1, 1, 1), activation="elu")(X_input)
    # conv_2 = Conv3D(filters=64, kernel_size=(1, 1, 1), padding="same", strides=(1, 1, 1), activation="elu")(conv_1)
    # conv_3 = Conv3D(filters=128, kernel_size=(1, 1, 1), padding="same", strides=(1, 1, 1), activation="elu")(conv_2)
    
    # conv_3 = Conv3D(filters=filters, kernel_size=kernel_size, padding="same", strides=(1, 1, 1), activation="elu")(X_input)
    shape = X_input.get_shape().as_list()
    
    pool_2_flat = Reshape([shape[1], shape[2]*shape[3]*shape[4]])(X_input)
    fc = Dense(recurrent_units, activation="elu")(pool_2_flat)
    fc_drop = Dropout(dropout_prob)(fc)
    
    lstm_in = Reshape([10, recurrent_units])(fc_drop)
    # lstm_1 = LSTM(units=1024, return_sequences=True, unit_forget_bias=True, dropout=dropout_prob)(lstm_in)
    # rnn_output = LSTM(units=1024, return_sequences=False, unit_forget_bias=True)(lstm_1)
    # lstm_1 = GRU(units=1024, return_sequences=True, dropout=dropout_prob)(lstm_in)
    rnn_output = Bidirectional(GRU(units=recurrent_units, return_sequences=False, dropout=dropout_prob, recurrent_dropout=dropout_prob))(lstm_in)
    
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
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=300, shuffle=False, validation_data=(X_valid, y_valid), callbacks=[cp_callback])
training_end_time = datetime.now()

model.save(results_path + "/model/model.h5")

with open(results_path + "/train_hist", "wb") as file:
    pickle.dump(history.history, file)

print("Training start date and time:", training_start_time)
print("Training end date and time:", training_end_time)
print("Training duration:", training_end_time - training_start_time)

# with open(results_path + "/readme.txt", "w") as file:
#     file.write("Training data: first 75%, 4 channels\n")
#     file.write("Validation data: last 25%, 4 channels\n")
#     file.write("Training start time: " + str(training_start_time) + "\n")
#     file.write("Training end time: " + str(training_end_time) + "\n")
#     file.write("Training duration: " + str(training_end_time - training_start_time) + "\n")

from contextlib import redirect_stdout

with open(results_path + '/model_summary.txt', 'w') as file:
    with redirect_stdout(file):
        model.summary()
