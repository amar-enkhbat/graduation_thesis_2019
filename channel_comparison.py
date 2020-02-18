import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import matplotlib; matplotlib.rc('font', family='TakaoPGothic')
# sns.set_context("talk")
# sns.set()
sns.set_context("paper", font_scale=1.5)

n_channels = [4, 8, 14, 32, 64]
val_accs = [0.528, 0.7, 0.833, 0.923, 0.978]

plt.figure(figsize=(10, 7))
plt.plot(n_channels, val_accs, marker="o")
for i, j in zip(n_channels, val_accs):
    plt.annotate(str(j), xy=(i-2, j+0.03))
plt.xticks(n_channels)
plt.grid()
plt.ylim(0, 1.05)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel("チャンネル数")
plt.ylabel("検証正解率")
plt.savefig("./images/channels_comparison.png", bbox_inches="tight")
plt.show()