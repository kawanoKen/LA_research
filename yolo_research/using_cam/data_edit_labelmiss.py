import os
import random
import sys
import numpy as np
from scipy.stats import norm

class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass  # 必須（Python 3 以降のバッファリング対応）

# === 設定 ===
log_path = 'rename_log.txt'
sys.stdout = Logger(log_path)  # print出力をファイルにも出力

# ラベルファイルが格納されているディレクトリ
label_dir = "/work/kawano/LA/datasets/african-wildlife/train/labels"
edit_label_dir = "/work/kawano/LA/datasets/african-wildlife_noise_labelmiss/train/labels"

def calculate_iou_xywh(box1, box2):
    """
    xywh形式の2つのバウンディングボックス間のIoU (Intersection over Union) を計算します。

    Args:
        box1 (list or tuple or np.ndarray): 1つ目のバウンディングボックス [x, y, w, h]
        box2 (list or tuple or np.ndarray): 2つ目のバウンディングボックス [x, y, w, h]

    Returns:
        float: 2つのバウンディングボックスのIoUの値。0から1の間の値を取ります。
               バウンディングボックスが存在しない場合や、Unionの面積が0の場合は0を返します。
    """

    # 入力形式の確認（簡易的）
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError("Input boxes must be in [x, y, w, h] format (length 4).")

    # xywh形式から xmin, ymin, xmax, ymax 形式に変換
    # box: [x_c, y_c, w, h]
    # x_min = x_c - w / 2
    # y_min = y_c - h / 2
    # x_max = x_c + w / 2
    # y_max = y_c + h / 2
    box1_xmin = box1[0] - box1[2] / 2
    box1_ymin = box1[1] - box1[3] / 2
    box1_xmax = box1[0] + box1[2] / 2
    box1_ymax = box1[1] + box1[3] / 2

    box2_xmin = box2[0] - box2[2] / 2
    box2_ymin = box2[1] - box2[3] / 2
    box2_xmax = box2[0] + box2[2] / 2
    box2_ymax = box2[1] + box2[3] / 2

    # 共通部分 (Intersection) の座標を計算
    inter_xmin = max(box1_xmin, box2_xmin)
    inter_ymin = max(box1_ymin, box2_ymin)
    inter_xmax = min(box1_xmax, box2_xmax)
    inter_ymax = min(box1_ymax, box2_ymax)

    # 共通部分の幅と高さを計算
    # max(0, ...) とすることで、共通部分が存在しない場合に負の値にならないようにする
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)

    # 共通部分の面積 (Intersection Area) を計算
    inter_area = inter_width * inter_height

    # 各バウンディングボックスの面積を計算
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    # 結合部分 (Union Area) を計算
    # Union Area = Area(box1) + Area(box2) - Intersection Area
    union_area = box1_area + box2_area - inter_area

    # Union Areaが0の場合は、IoUも0とする (分母が0になるのを防ぐ)
    if union_area <= 0:
        return 0.0

    # IoUを計算
    iou = inter_area / union_area

    return iou

def noise_uniform(proportioin):
    n = random.random()
    if n > proportioin: return 0
    # return -1 to 1
    else: return 2 * n / proportioin- 1

def noise_gausssian(proportioin, ):
    np.random.normal(
    loc   = 0,      # 平均
    scale = 1,      # 標準偏差
    )

rng = np.random.default_rng()

def delete_files_not_starting_with_1_random(label_dir,):
    for file_name in os.listdir(label_dir):
        # ファイル名が '2' で始まらない場合
        if (not file_name.startswith('2')) and random.random() < 0.95:
            file_path = os.path.join(label_dir, file_name)
            try:
                os.remove(file_path)
                print(f"削除しました: {file_name}")
            except Exception as e:
                print(f"削除エラー: {file_name} → {e}")
        else:
            print(f"保持: {file_name}")

delete_files_not_starting_with_1_random(edit_label_dir)

# 実行
