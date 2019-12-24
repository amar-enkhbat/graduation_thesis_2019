import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
"keras_2019_12_17_15_04"
results_paths = ["keras_2019_12_12_20_51", "keras_2019_12_13_22_41",  
"keras_2019_12_18_14_24", "keras_2019_12_20_21_32", "keras_2019_12_21_17_21",
"keras_2019_12_22_19_53", "keras_2019_12_23_12_11"]

for results_path in results_paths:
    results_path = "./results/" + results_path 
    with open(results_path + "/train_hist", "rb") as file:
        history = pickle.load(file)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()