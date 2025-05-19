from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import math
import numpy as np
import torch    
import cv2
from ultralytics.utils.instance import Instances, Bboxes
from ultralytics.data.augment import LetterBox
import os
import glob
import torchvision
import json
import os
from pathlib import Path


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    conf: float = 1.0               # GT は conf=1.0, 予測はモデル信頼度

# ──────────────────────────────────────────────────────────────
# 2. 定数（cleanlab デフォルト）
# ──────────────────────────────────────────────────────────────
ALPHA   = 0.9              # IoU と距離類似の混合係数
LOW_P   = 0.5              # BadLoc/Swap で無視する低信頼閾値
HIGH_P  = 0.95             # Overlooked 判定に使う高信頼閾値
TEMP    = 0.1              # soft-min の温度
EUC_FAC = 0.1              # 中心距離 → exp() 係数
DUP_IOU = 0.95             # 異クラス GT が重なるときの IOU
TINY    = 1e-6

# ──────────────────────────────────────────────────────────────
# 3. 基本演算
# ──────────────────────────────────────────────────────────────
def iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1); y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2); y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / ((a.x2-a.x1)*(a.y2-a.y1) +
                    (b.x2-b.x1)*(b.y2-b.y1) - inter + TINY)

def center_similarity(a: Box, b: Box) -> float:
    cx_a, cy_a = (a.x1+a.x2)/2, (a.y1+a.y2)/2
    cx_b, cy_b = (b.x1+b.x2)/2, (b.y1+b.y2)/2
    # exp(-dist * factor) を (1-α) 重み側に使う
    return math.exp(-math.hypot(cx_a-cx_b, cy_a-cy_b) * EUC_FAC)

def similarity(a: Box, b: Box) -> float:
    return ALPHA * iou(a, b) + (1 - ALPHA) * (1 - center_similarity(a, b))

def softmin1d(arr: Sequence[float], temp: float = TEMP) -> float:
    arr = np.array([x for x in arr if not math.isnan(x)])
    if arr.size == 0:
        return 1.0
    w = np.exp(-arr * temp)
    w /= w.sum() + TINY
    return float(np.dot(arr, w))

# ──────────────────────────────────────────────────────────────
# 4-A. Overlooked Box Scores（予測ごと）
# ──────────────────────────────────────────────────────────────
def overlooked_scores(labels: Sequence[Box], preds: Sequence[Box]) -> List[float]:
    """
    長さ len(preds) のリストを返す。低いほど「GT に無いのに高信頼で検出」の疑い。
    """
    out = []
    for p in preds:
        # ① 高信頼でなければスキップ（NaN 扱い）
        if p.conf < HIGH_P:
            out.append(float("nan"))
            continue
        # ② IoU>0 の GT があれば見落としではない
        if any(iou(p, g) > 0 for g in labels if g.cls == p.cls):
            out.append(float("nan"))
            continue
        # ③ 同クラス GT が 1 つも無い
        same_cls = [g for g in labels if g.cls == p.cls]
        if not same_cls:
            # cleanlab: sim* は非ゼロ最小値。簡易版として 0.
            out.append((1 - p.conf) * 0.0)
        else:
            out.append(max(similarity(g, p) for g in same_cls))
    return out

# ──────────────────────────────────────────────────────────────
# 4-B. BadLoc Box Scores（GT ごと）
# ──────────────────────────────────────────────────────────────
def badloc_scores(labels: Sequence[Box], preds: Sequence[Box]) -> List[float]:
    """
    長さ len(labels) のリスト。低いほど「GT と位置ズレが大きい」。
    """
    out = []
    for g in labels:
        # 同クラスで conf>=LOW_P の予測のみ対象
        cand = [p for p in preds if p.cls == g.cls and p.conf >= LOW_P]
        if not cand:
            # out.append(float("nan")) #公式では1扱い
            out.append(1.0)
            continue
        out.append(max(similarity(g, p) for p in cand))
    return out

# ──────────────────────────────────────────────────────────────
# 4-C. Swapped Box Scores（GT ごと）
# ──────────────────────────────────────────────────────────────
def swapped_scores(labels: Sequence[Box], preds: Sequence[Box]) -> List[float]:
    """
    長さ len(labels) のリスト。低いほど「別クラス予測に置き換わっている」疑い。
    """
    # “重複 GT 別ラベル” 判定
    dup_penalty = [any(i != j and iou(a, b) >= DUP_IOU and a.cls != b.cls
                       for j, b in enumerate(labels))
                   for i, a in enumerate(labels)]

    out = []
    for i, g in enumerate(labels):
        cand = [p for p in preds if p.cls != g.cls and p.conf >= LOW_P]
        if not cand:
            out.append(1.0)
            continue
        worst = 1 - max(similarity(g, p) for p in cand)
        out.append(worst * (0.5 if dup_penalty[i] else 1.0)) #公式では重複(閾値以上のIoU)のあるGtは最低スコアを与える.
    return out


def convert_elephant_zebra(id):
    if id == 0:
        return 20
    elif id == 1:
        return 22



# Load a model
model = YOLO("yolo11x.yaml")  # build a new model from YAML
model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights


img_exts=('.jpg', '.jpeg', '.png')
quority = {}
dir = "/work/kawano/LA/datasets/african-wildlife_noise/"
for i in ["train", "valid"]:
    img_dir = os.path.join(dir, i, "images")
    ann_dir = os.path.join(dir, i, "labels")
    ann_paths = glob.glob(os.path.join(ann_dir, '*.txt'))
    letterbox = LetterBox(auto = True)
    
    for ann_path in ann_paths:
            base = os.path.splitext(os.path.basename(ann_path))[0]

            # 2. 画像ファイルを拡張子候補から探す
            img_path = None
            for ext in img_exts:
                candidate = os.path.join(img_dir, base + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            print(img_path)
            img_file = Path(img_path).stem
            quority[f"{img_file}"]={}
            img = cv2.imread(img_path)
            img = img.astype(np.float32)
            # if img_path is None:
            #     print(f"[警告] 画像が見つかりません: {base} に対応する{img_exts}のいずれか")
            #     continue
            bboxes = []
            class_ids = []
            # 3. アノテーション読み込み
            for line in open(ann_path, 'r'):
                line = list(map(float, line.strip().split()))
                class_id = int(line[0])
                bbox = line[1:5]
                bboxes.append(bbox)
                class_ids.append(convert_elephant_zebra(class_id))
                #edit.append(class_id)
            instance = Instances(bboxes=np.array(bboxes), segments = np.zeros((2,2)), bbox_format='xywh', normalized=True)
            instance._bboxes.convert("xyxy")
            label_dict = {"img": img, "instances": instance}

            letterboxed = letterbox(labels=label_dict)
            img = letterboxed["img"]
            gt_bbox = letterboxed["instances"].bboxes.tolist()

            # 整形
            img_letterboxed = letterbox(image = img)
            # tensorに直す
            img_tensor = torch.from_numpy(img_letterboxed).permute(2, 0, 1).unsqueeze(0)
            img_tensor /= 255

            #推論
            result = model(img_tensor)[0].boxes
            bbox_pre = result.xyxy.tolist()
            conf_pre = result.conf.tolist()
            cls_pre = result.cls.tolist()
            labels = [Box(gt_bbox[i][0], gt_bbox[i][1], gt_bbox[i][2], gt_bbox[i][3], class_ids[i])for i in range(len(gt_bbox))]
            pred_boxes = [Box(bbox_pre[i][0], bbox_pre[i][1], bbox_pre[i][2], bbox_pre[i][3], cls_pre[i], conf_pre[i]) for i in range(len(bbox_pre))]

            badloc = badloc_scores(labels, pred_boxes)
            swapped = swapped_scores(labels, pred_boxes)
            #overlooked = overlooked_scores(labels, pred_boxes)
            quority[f"{img_file}"]["badloc_scores"] = badloc
            quority[f"{img_file}"]["swapped_scores"] = swapped
            #quority[f"{img_file}"]["overlooked_scores"] = overlooked

with open("quority_african-wildlife_noise/african_wildlife_elephant_objectlab.json", "w") as f: 
    json.dump(quority, f)


