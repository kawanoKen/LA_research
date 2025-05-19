import os
import random
import sys
import numpy as np
from scipy.stats import norm
import os
import glob

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
edit_label_dir = "/work/kawano/LA/datasets/african-wildlife_noise/train/labels"

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


def delete_files_not_starting_with_1_3(label_dir,):
    for file_name in os.listdir(label_dir):
        # ファイル名が '2' で始まらない場合
        if file_name.startswith('1') or file_name.startswith('3'):
            file_path = os.path.join(label_dir, file_name)
            try:
                os.remove(file_path)
                print(f"削除しました: {file_name}")
            except Exception as e:
                print(f"削除エラー: {file_name} → {e}")
        else:
            print(f"保持: {file_name}")

def edit_yolo_labels(label_dir, edit_label_dir):
    for file_name in os.listdir(label_dir):

        if not file_name.endswith('.txt'):
            continue

        file_path = os.path.join(label_dir, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        new_IoUs = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            parts = map(float, parts)
            parts = list(parts)
            if(parts[0] == 1):
                parts[0] = 0
            elif (parts[0] == 3):
                parts[0] = 1
            else:
                continue
            new_parts = [parts[0]]
            parts_xyxy = [parts[0], parts[1] - parts[3]/2, parts[2] - parts[4]/2, parts[1] + parts[3]/2, parts[2] + parts[4]/2]
            for i in range(4):
                    new =float(parts_xyxy[i+1] + noise_uniform(0.1) * 0.3)
                    new_parts.append(new)
            new_parts_xywh = [new_parts[0], round((new_parts[3] + new_parts[1])/2, 6) , round((new_parts[4] + new_parts[2])/2, 6), round(new_parts[3] - new_parts[1], 6), round(new_parts[4] - new_parts[2], 6)]
            IoU = calculate_iou_xywh(parts[1:], new_parts_xywh[1:])
            truth = (0<=new_parts[1]) & (new_parts[1]<=new_parts[3]) & (new_parts[3]<=1) & (0<=new_parts[2]) & (new_parts[2]<=new_parts[4]) & (new_parts[4]<=1)
            if truth:
                new_lines.append(' '.join(list(map(str, new_parts_xywh)) + ["\n"]))
                print(f"{file_path}を編集, IoU = {IoU}")
                new_IoUs.append(f"{round(IoU, 6)}\n")
            else:
                new_lines.append(' '.join(list(map(str, parts)) + ["\n"]))
                print(f"{file_path}を編集, IoU = {1}") 
                new_IoUs.append("1\n")
        
        generate_random = random.random()
        if generate_random < 0.1:
            x, X = random.random(), random.random()
            y, Y = random.random(), random.random()
            x, X = min(x, X), max(x, X)
            y, Y = min(y, Y), max(y, Y)
            new_parts_xywh = [int(generate_random<0.05), round((x + X)/2, 6) , round((y+Y)/2, 6), round(X-x, 6), round(Y-y, 6)]
            new_lines.append(' '.join(list(map(str, new_parts_xywh)) + ["\n"]))
            new_IoUs.append("0\n")

        edit_file_path = os.path.join(edit_label_dir, file_name)
        with open(edit_file_path, 'w') as f:
            f.writelines(new_lines)

        edit_IoU_path = os.path.join("/work/kawano/LA/yolo_research/IoU_african-wildlife_noise", file_name)
        with open(edit_IoU_path, 'w') as f:
            f.writelines(new_IoUs)

    print("ラベルの変換が完了しました。")

breakpoint()
edit_yolo_labels(label_dir, edit_label_dir)
delete_files_not_starting_with_1_3(edit_label_dir)

# 実行
