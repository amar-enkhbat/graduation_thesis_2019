#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

with open("./training_keras_hist_2019_11_21/train_hist", "rb") as file:
    history = pickle.load(file)

plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

output_dir 	= "conv_3l_win_10_108_fc_rnn2_fc_1024_N_summary_075_train_posibility_roc"
output_file = "conv_3l_win_10_108_fc_rnn2_fc_1024_N_summary_075_train_posibility_roc"

dataset_dir = "./preprocessed_dataset/"

with open(dataset_dir+"1_108_shuffle_dataset_3D_win_10.pkl", "rb") as fp:
  	dataset = pickle.load(fp)
with open(dataset_dir+"1_108_shuffle_labels_3D_win_10.pkl", "rb") as fp:
  	labels = pickle.load(fp)
dataset = dataset.reshape(len(dataset), 10, 10, 11, 1)
print("Dataset shape:", dataset.shape)
print("Labels shape:", labels.shape)

def one_hot_encoder(labels):
    return np.asarray(pd.get_dummies(labels), dtype = np.int8)
labels = one_hot_encoder(labels)
print("One-hot-encoded labels:")
print(labels)

split = np.random.rand(len(dataset)) < 0.75

X_train = dataset[split]
y_train = labels[split]

X_test = dataset[~split] 
y_test = labels[~split]

print("Train dataset shape:", X_train.shape)
print("Train label shape:", y_train.shape)
print("Test dataset shape:", X_test.shape)
print("Test label shape:", y_test.shape)


# # Model load and predict

dropout_prob = 0.5
n_labels = y_train.shape[1]
training_epochs = 10
batch_size = 300
learning_rate = 1e-4

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Conv2D, Conv3D
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

model = model(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))

opt = Adam(lr=learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

checkpoint_path = "training_keras_2019_11_21/cp.ckpt"
model.load_weights(checkpoint_path)

y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

y_pred_bool = one_hot_encoder(y_pred_bool)

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

precision = dict()
recall = dict()
for i in range(5):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_pred_bool[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_bool))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred_bool))
