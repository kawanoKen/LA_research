from ultralytics import YOLO
from ultralytics.utils.instance import Instances, Bboxes
import torch    
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM, AblationCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
import torchvision
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.data.dataset import YOLODataset
import matplotlib.pyplot as plt
import os
import glob
import csv


    
model = YOLO('yolov8n.pt')  # or yolov8s.pt, etc.d
class yolo_target_score:
    def __init__(self, bounding_boxes, labels, threshold = 0.5):
        self.bounding_boxes = bounding_boxes
        self.labels = labels
        self.threshold = threshold
    def __call__(self, output):
            target = torch.Tensor([0])
            output_cls = output[0][4:].T
            output_bbox = Bboxes(output[0][:4].T, format = "xywh")
            output_bbox.convert("xyxy")
            output_bbox = output_bbox.bboxes.cuda()
            output_cls = output_cls.cuda()
            target = target.cuda()
            
            for box, label in zip(self.bounding_boxes, self.labels):
                box = torch.Tensor(box[None, :])
                if torch.cuda.is_available():
                    box = box.cuda()
                ious = torchvision.ops.box_iou(box, output_bbox)[0]
                indexes = (ious > 0.8) & (output_cls.argmax(1) == label)
                score = (ious[indexes] + output_cls.max(1).values[indexes]).sum()
                target = target + score
            return target
    
            # #出力がpredictだった時用の処理
            # target = torch.Tensor([0])
            # output_bbox = output.boxes.xyxy
            # output_cls = output.boxes.cls
            # output_conf = output.boxes.conf

            # for box, label in zip(self.bounding_boxes, self.labels):
            #     box = torch.Tensor(box[None, :])
            #     if torch.cuda.is_available():
            #         target = target.cuda()
            #         output_bbox = output_bbox.cuda()
            #         output_cls = output_cls.cuda()
            #         output_conf = output_conf.cuda()
            #         box = box.cuda()

            #     ious = torchvision.ops.box_iou(box, output_bbox)
            #     index = ious.argmax()
            #     if ious[0, index] > self.threshold and output_cls[index] == label:
            #         score = ious[0, index] + output_conf[index]
            #         target = target + score
            # return target

target_layers = [model.model.model[i] for i in range(10)] + [model.model.model[i] for i in [12, 15, 18, 21]]

img_exts=('.jpg', '.jpeg', '.png')
img_dir = '/work/kawano/LA/datasets/coco/images/val2017'


ann_dirs = ["/work/kawano/LA/yolo_research/coco_test/elephant"]
# ann_dirs = ["/work/kawano/LA/yolo_research/coco_test/bicycle"]
# ann_dirs = ["/work/kawano/LA/yolo_research/coco_test/car"]
max_grads_cls = []
for ann_dir in ann_dirs:
    ann_paths = glob.glob(os.path.join(ann_dir, '*.txt'))
    max_grads_pairs = []
    for ann_path in ann_paths:
        base = os.path.splitext(os.path.basename(ann_path))[0]

        # 2. 画像ファイルを拡張子候補から探す
        img_path = None
        for ext in img_exts:
            candidate = os.path.join(img_dir, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        # if img_path is None:
        #     print(f"[警告] 画像が見つかりません: {base} に対応する{img_exts}のいずれか")
        #     continue

        # 3. アノテーション読み込み
        with open(ann_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        # パース例: "0 0.512345 0.432100 0.123456 0.234567"
        cls_ids = []
        bboxes = []
        for l in lines:
            cls_id, x, y, w, h = l.split()
            cls_ids.append(int(cls_id))
            bboxes.append([float(x), float(y), float(w), float(h)])

        img = cv2.imread(img_path)
        img = img.astype(np.float32)

        instance = Instances(bboxes=np.array(bboxes), segments = np.zeros((2,2)), bbox_format='xywh', normalized=True)
        instance.classes = np.array(cls_ids)
        label_dict = {"img": img, "instances": instance}
        letterbox = LetterBox(auto = True)
        
        letterboxed = letterbox(labels=label_dict)
        img = letterboxed["img"]
        gt_bbox = letterboxed["instances"].bboxes
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img_tensor /= 255
        img_tensor.requires_grad = True
        target = [yolo_target_score(gt_bbox, instance.classes)]
        
        cam = GradCAM(model.model, target_layers)


        cam.model.zero_grad()
        output = cam.activations_and_grads(img_tensor)
        loss = sum([target(output) for target, output in zip(target, output)])
        loss.backward(retain_graph=True)
        grads = cam.activations_and_grads.gradients

        max_grads = []
        for grad in grads:
            b, c, w, h = grad.shape[0], grad.shape[1], grad.shape[2], grad.shape[3]
            max_grad = grad.view(b, c, w * h).max(2)[0][0]

            max_grads.append(max_grad)
        max_grads_pairs.append(max_grads)
    max_grads_cls.append(max_grads_pairs)
import json

grad_layer = []
for i in range(len(target_layers)):
    grad_sum = torch.zeros(len(max_grads_cls[0][0][i]))
    for j in range(len(max_grads_cls[0])):
       grad_sum += max_grads_cls[0][j][i]
    grad_layer.append(torch.topk(grad_sum, round(grad_sum.shape[0] / 4)))

print([grad.indices.tolist() for grad in grad_layer])

# with open("/work/kawano/LA/yolo_research/coco_test/grad_elephant.csv", "w") as f:
#     json.dump(max_grads_cls, f)

