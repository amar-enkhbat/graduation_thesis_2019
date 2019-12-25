import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
results_dirs = os.listdir("./results/")

# for results_dir in results_dirs:
#     history_path = "./results/" + results_dir 
#     with open(history_path + "/train_hist", "rb") as file:
#         history = pickle.load(file)
#     plt.plot(history['acc'])
#     plt.plot(history['val_acc'])
#     plt.title(results_dir)
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.grid()
#     plt.ylim(0, 1)
#     plt.savefig("./images/" + results_dir)
#     plt.show()

for results_dir in results_dirs:
    history_path = "./results/" + results_dir 
    with open(history_path + "/train_hist", "rb") as file:
        history = pickle.load(file)
    plt.plot(history['acc'], label=(results_dir + "_train"))
    plt.plot(history['val_acc'], label=(results_dir + "_valid"))
    plt.title("Results")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend()
    plt.grid()
    plt.ylim(0, 1)
plt.savefig("./images/" + "matome")
plt.show()