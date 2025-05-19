import numpy as np, hdbscan, umap
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import json, numpy as np


with open("african_wildlife_elephant_toplayer.json") as f:
    ann_info = json.load(f)

# 1. データ読み込み (564, D)
X = np.array(ann_info["mean_feature_using"]).astype('float32')

feature_argsort = X.argsort(axis = 0)
edit = np.array(ann_info["edit"])
edit_sort = np.array([[(edit[int(index)].tolist())  for index in index_layer]for index_layer in feature_argsort])

feature_norm = X.sum(1)
feature_norm_argsort = feature_norm.argsort()
edit_norm_sort = np.array([(edit[int(index)].tolist())  for index in feature_norm_argsort])

breakpoint()

import matplotlib.pyplot as plt # Matplotlibのpyplotモジュールをインポート

fig, axes = plt.subplots(nrows=14, ncols=1, figsize=(40, 40))
x = np.arange(len(edit_sort))
for i in range(14):
    ax1 = axes[i] # 1つ目の Axes オブジェクトにアクセス
    ax1.bar(x, edit_sort[:, i], width=1.0, color='skyblue') # ax1 に対して bar メソッドを呼び出し
    ax1.set_title(f"{i+1} layer") # ax1 のタイトルを設定
    ax1.set_xlabel("特徴量ソート")
    ax1.set_ylabel('アノテーション信用度')
# グラフを表示
plt.show()
plt.savefig("data_generate")

# 2. 前処理（距離希薄化対策）
X = normalize(X)               # L2=1 → cosine距離 ≈ L2
X = PCA(0.9).fit_transform(X)  # 分散90%で次元圧縮

# 3. UMAP で低次元化 (可視化&クラスタ容易化)
um = umap.UMAP(n_components=10, metric='euclidean').fit_transform(X)

# 4. HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=15,
                            metric='euclidean',
                            prediction_data=True).fit(um)

labels      = clusterer.labels_          # -1 は外れ値
probability = clusterer.probabilities_   # “クラスタらしさ” 0–1

breakpoint()
# 5. 結果集計
n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers  = (labels == -1).sum()
print(f"clusters={n_clusters}, outliers={n_outliers}")