import numpy as np, hdbscan, umap
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import json, numpy as np


with open("african_wildlife_elephant_pre.json") as f:
    ann_info = json.load(f)

# 1. データ読み込み (564, D)
#X = np.array(ann_info["mean_feature_using"]).astype('float32')
conf = np.array(ann_info["conf"]).astype('float32')
IoU = np.array(ann_info["IoU"]).astype('float32')
X = conf + IoU
feature_argsort = X.argsort(axis = 0)
edit = np.array(ann_info["edit"])
edit_sort = np.array([(edit[int(index)].tolist()) for index in feature_argsort])

# feature_norm = X.sum(1)
# feature_norm_argsort = feature_norm.argsort()
# edit_norm_sort = np.array([(edit[int(index)].tolist())  for index in feature_norm_argsort])

breakpoint()

import matplotlib.pyplot as plt # Matplotlibのpyplotモジュールをインポート
x = np.arange(len(edit_sort))
plt.bar(x, edit_sort, width=1.0)
plt.ylabel("wrong")
plt.show()
plt.savefig("data_noise_pre")
