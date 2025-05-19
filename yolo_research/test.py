from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

# モデル読み込み（軽いモデル推奨）
model = YOLO('yolov8n.pt')  # or yolov8s.pt, etc.
model.eval()

# # モデルのバックボーンを取得
backbone = model.model.model


# def preprocess_image(img_path, target_size=640):
#     """
#     Ultralytics YOLOv8 に入力可能な形式に画像を変換する関数
#     - img: BGR または RGB の画像（NumPy配列）
#     - target_size: 入力解像度（例：640）
#     - return: 前処理済みの torch.Tensor（1, 3, H, W）
#     """
#     img = cv2.imread(img_path)
#     # 1. Resize with letterbox (padding to preserve aspect ratio)
#     img_letterbox = LetterBox(new_shape=target_size)(image=img)

#     # 2. Convert BGR to RGB if needed
#     if img_letterbox.shape[2] == 3:  # 確実にRGB形式へ
#         img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
#     else:
#         img_rgb = img_letterbox

#     # 3. Convert to float32 and normalize [0, 1]
#     img_norm = img_rgb.astype(np.float32) / 255.0

#     # 4. HWC → CHW
#     img_transposed = np.transpose(img_norm, (2, 0, 1))

#     # 5. Convert to torch.Tensor
#     img_tensor = torch.from_numpy(img_transposed)

#     # 6. Add batch dimension
#     img_tensor = img_tensor.unsqueeze(0)  # shape: (1, 3, H, W)

#     return img_tensor

# 畳み込み層の出力を格納するリスト
conv_outputs = {}
conv_inputs = {}
cnn_w = []

# フック関数定義
def get_activation(name):
    def hook(model, input, output):
            conv_inputs[name] = input
            if type(output) is tuple:
                conv_outputs[name] = output
            else:
                conv_outputs[name] = output[0].detach().clone()
    return hook

# 全ての畳み込み層にフック登録
for idx, layer in enumerate(backbone):
    # cnn_w.append(layer.conv.weight[:, 0, :,:])
    layer.register_forward_hook(get_activation(f"{idx}-{layer.__class__.__name__}"))

img_path = "/work/kawano/LA/yolo_research/img/caar.jpeg"
with torch.no_grad():
    result = model(img_path)  # forward pass

breakpoint()
# for k in conv_outputs:
#     if type(conv_outputs[k]) is tuple:
#         print(f"{k}の出力形式は({conv_outputs[k][0].shape}, {conv_outputs[k][1][0].shape}, {conv_outputs[k][1][1].shape})")
        

#     else:
#         output = conv_outputs[k]
#         print(f"{k}の出力形式は{output.shape}")

#         plt.figure(figsize=(3 * output.shape[0], 3), dpi=300)
#         for i in range(output.shape[0]):
#             plt.subplot(1, output.shape[0], i + 1)
#             plt.imshow(output[i].to('cpu'), cmap='viridis')
#             plt.axis('off')
#             plt.title(f'Channel {i}')
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(f"img/{k}.png")