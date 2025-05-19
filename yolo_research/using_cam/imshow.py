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

model("/work/kawano/LA/datasets/african-wildlife_edit/train/images/2 (336).jpg", save=True)

