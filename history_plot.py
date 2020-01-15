import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
results_dirs = os.listdir("./results/")


for results_dir in results_dirs:
    history_path = "./results/" + results_dir 
    with open(history_path + "/train_hist", "rb") as file:
        history = pickle.load(file)
    plt.plot(history['acc'], label="Training")
    plt.plot(history['val_acc'], label="Validation")
    plt.title(results_dir)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.grid(0.1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    plt.savefig("./images/" + results_dir)
    plt.clf()
    # plt.show()

# for results_dir in results_dirs:
#     history_path = "./results/" + results_dir 
#     with open(history_path + "/train_hist", "rb") as file:
#         history = pickle.load(file)
#     plt.plot(history['acc'], label=(results_dir + "_train"))
#     plt.plot(history['val_acc'], label=(results_dir + "_valid"))
#     plt.title("Results")
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     # plt.legend()
#     plt.grid()
#     plt.ylim(0, 1)
# # plt.savefig("./images/" + "matome")
# plt.show()

# results_dirs = ["keras_2019_12_26_18_26", "keras_2019_12_26_20_17", "keras_2019_12_27_02_25"]
# results_dirs = ["keras_2019_12_26_18_26"]
# keras_2019_12_26_18_26: conv 32 filters, gru 128 units
# keras_2019_12_26_20_17: conv 64 filters, gru 128 units
# keras_2019_12_27_02_25: conv 128 filters, gru 128 units
# keras_2019_12_27_15_02: conv 32 filters, gru 512 units
# for results_dir in results_dirs:
#     history_path = "./results/" + results_dir 
#     with open(history_path + "/train_hist", "rb") as file:
#         history = pickle.load(file)
#     plt.plot(history['acc'], label=(results_dir + "_train"), linestyle="-")
#     plt.plot(history['val_acc'], label=(results_dir + "_valid"), linestyle=":")
#     plt.title("Results")
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend()
#     plt.grid()
#     # plt.ylim(0, 1)
# plt.savefig("./images/" + "matome_cnn_reduction")
# plt.show()