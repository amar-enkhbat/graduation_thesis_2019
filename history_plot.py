import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import os

import matplotlib; matplotlib.rc('font', family='TakaoPGothic')
# sns.set()
# sns.set_context("talk")
sns.set_context("notebook", font_scale=1.5)

results_dirs = os.listdir("./results/")
results_dirs = sorted(results_dirs)
print(results_dirs)

for results_dir in results_dirs:
    history_path = "./results/" + results_dir 
    with open(history_path + "/train_hist", "rb") as file:
        history = pickle.load(file)
    plt.figure(figsize=(10, 7))
    plt.plot(history['acc'], color="C0", label="学習")
    plt.plot(history['val_acc'], color="C1", label="検証")
    max_val_index = np.argmax(history["acc"])
    max_val = history["acc"][max_val_index]
    plt.scatter(max_val_index, history["acc"][max_val_index], marker="x", color="blue", label=("最大学習正解率: " + str(max_val.round(3))))
    max_val_index = np.argmax(history["val_acc"])
    max_val = history["val_acc"][max_val_index]
    plt.scatter(max_val_index, history["val_acc"][max_val_index], marker="x", color="red", label=("最大検証正解率: " + str(max_val.round(3))))
    plt.title(results_dir)
    plt.ylabel('正解率')
    plt.xlabel('エポック数')
    plt.legend(loc='lower right')
    plt.grid()
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1.05)
    # plt.show()
    plt.savefig("./images/" + results_dir, bbox_inches="tight")
    plt.clf()

# for results_dir in results_dirs:
#     history_path = "./results/" + results_dir 
#     with open(history_path + "/train_hist", "rb") as file:
#         history = pickle.load(file)
#     plt.plot(history['acc'], color="C0", label="Training")
#     plt.plot(history['val_acc'], color="C1", label="Validation")
#     max_val_index = np.argmax(history["acc"])
#     max_val = history["acc"][max_val_index]
#     # plt.annotate("Max Acc", xy=(max_val_index, history["val_acc"][max_val_index]), xycoords="data", label=("Max Val Acc: " + str(max_val.round(2))))
#     plt.scatter(max_val_index, history["acc"][max_val_index], marker="x", color="blue", label=("Max Train Acc: " + str(max_val.round(3))))
#     max_val_index = np.argmax(history["val_acc"])
#     max_val = history["val_acc"][max_val_index]
#     # plt.annotate("Max Acc", xy=(max_val_index, history["val_acc"][max_val_index]), xycoords="data", label=("Max Val Acc: " + str(max_val.round(2))))
#     plt.scatter(max_val_index, history["val_acc"][max_val_index], marker="x", color="red", label=("Max Val Acc: " + str(max_val.round(3))))
#     plt.title("Results")
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     # plt.legend()
#     plt.grid()
#     plt.ylim(0, 1)
# plt.savefig("./images/" + "matome", bbox_inches="tight")
# # plt.show()

# results_dirs = ["keras_2019_12_12_20_51", "keras_2020_01_23_11_56"]

# plt.figure(figsize=(10, 7))

# history_path = "./results/" + "keras_2019_12_12_20_51"
# with open(history_path + "/train_hist", "rb") as file:
#     history = pickle.load(file)
# plt.plot(history['acc'], label=("jikken1_train"), linestyle="-", color="C1")
# plt.plot(history['val_acc'], label=("jikken1_valid"), linestyle="-.", color="C1")
# max_val_index = np.argmax(history["acc"])
# max_val = history["acc"][max_val_index]
# plt.scatter(max_val_index, history["acc"][max_val_index], marker="x", color="C1", label=("Max Train Acc: " + str(max_val.round(3))))
# max_val_index = np.argmax(history["val_acc"])
# max_val = history["val_acc"][max_val_index]
# plt.scatter(max_val_index, history["val_acc"][max_val_index], marker="o", color="C1", label=("Max Val Acc: " + str(max_val.round(3))))

# history_path = "./results/" + "keras_2020_01_23_11_56"
# with open(history_path + "/train_hist", "rb") as file:
#     history = pickle.load(file)
# plt.plot(history['acc'], label=("jikken2_train"), linestyle="-", color="C0")
# plt.plot(history['val_acc'], label=("jikken2_valid"), linestyle="-.", color="C0")
# max_val_index = np.argmax(history["acc"])
# max_val = history["acc"][max_val_index]
# plt.scatter(max_val_index, history["acc"][max_val_index], marker="x", color="C0", label=("Max Train Acc: " + str(max_val.round(3))))
# max_val_index = np.argmax(history["val_acc"])
# max_val = history["val_acc"][max_val_index]
# plt.scatter(max_val_index, history["val_acc"][max_val_index], marker="o", color="C0", label=("Max Val Acc: " + str(max_val.round(3))))

# plt.title("Results")
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(loc="lower right")
# plt.ylim(0, 1.05)
# plt.savefig("./images/" + "good_and_bad", bbox_inches="tight")
# plt.show()