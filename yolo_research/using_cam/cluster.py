from sklearn.cluster import KMeans

from ultralytics import YOLO
import torch
import os
import numpy as np

# モデル読み込み（軽いモデル推奨）
model = YOLO('yolov8n.pt')  # or yolov8s.pt, etc.
model.eval()

# # モデルのバックボーンを取得
backbone = model.model.model

conv_outputs = {}
conv_info =  np.array([])
# フック関数定義
def get_activation(name):
    def hook(model, input, output):
        conv_outputs[name] = output[0]
        np.concatenate([conv_info[:], output[0].sum((1, 2)).cpu().numpy()])
    return hook



# 全ての畳み込み層にフック登録
for idx, layer in enumerate(backbone):
    if idx in [15, 18, 21]:
        layer.register_forward_hook(get_activation(f"{idx}"))

# ダミー入力で実行（サイズはモデルに応じて調整）
img_dir = "/work/kawano/LA/yolo_research/dataset"
model("/work/kawano/LA/yolo_research/2 (135)_ann.jpg")

conv_outputs_all = {}
conv_info_all = np.array([])
files = []
with torch.no_grad():
    for file_name in os.listdir(img_dir):
        files.append(file_name)
        image_path = os.path.join(img_dir, file_name)
        conv_outputs = {}
        conv_info = np.array([])
        model(image_path)
        conv_outputs_all[file_name] = conv_outputs
        conv_info_all = np.concatenate([conv_info_all, conv_info], axis=0)
        breakpoint()
breakpoint()
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(conv_info_all)

y_kmeans = kmeans.predict(conv_info_all)

