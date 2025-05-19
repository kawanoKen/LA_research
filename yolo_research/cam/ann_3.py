from ultralytics import YOLO
import torch    
import cv2
import numpy as np
from ultralytics.utils.instance import Instances, Bboxes
from ultralytics.data.augment import LetterBox
import os
import glob
import torchvision


model = YOLO('yolov8n.pt')  # or yolov8s.pt, etc.d
model(torch.zeros(1, 3, 640, 640))  # or yolov8s.pt, etc.d



img_exts=('.jpg', '.jpeg', '.png')
img_dir = "/work/kawano/LA/datasets/african-wildlife_noise/train/images"
ann_dir = "/work/kawano/LA/datasets/african-wildlife_noise_generate/train/labels"
ann_dir = "/work/kawano/LA/datasets/african-wildlife_noise_labelmiss/train/labels"
ann_dir = "/work/kawano/LA/datasets/african-wildlife_noise/train/labels"
anninfo = {}
anninfo["ann_path"] = []
anninfo["ann_num"] = []
anninfo["edit"] = []
anninfo["conf"] = []
anninfo["IoU"] = []
features = []
max_features = []

ann_paths = glob.glob(os.path.join(ann_dir, '*.txt'))
m = 0
letterbox = LetterBox(auto = True)
for ann_path in ann_paths:
        m +=1
        print(f"\n {m}/268 個目のアノテーション{ann_path}\n")
        base = os.path.splitext(os.path.basename(ann_path))[0]

        # 2. 画像ファイルを拡張子候補から探す
        img_path = None
        for ext in img_exts:
            candidate = os.path.join(img_dir, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        
        img = cv2.imread(img_path)
        img = img.astype(np.float32)
        # if img_path is None:
        #     print(f"[警告] 画像が見つかりません: {base} に対応する{img_exts}のいずれか")
        #     continue
        bboxes = []
        class_ids = []
        edit = []
        # 3. アノテーション読み込み
        for line in open(ann_path, 'r'):
            line = list(map(float, line.strip().split()))
            class_id = line[0]
            bbox = line[1:5]
            bboxes.append(bbox)
            edit.append(line[5])
            #edit.append(class_id)
        instance = Instances(bboxes=np.array(bboxes), segments = np.zeros((2,2)), bbox_format='xywh', normalized=True)
        instance.classes = np.array(class_ids)
        label_dict = {"img": img, "instances": instance}

        letterboxed = letterbox(labels=label_dict)
        img = letterboxed["img"]
        gt_bbox = torch.tensor(letterboxed["instances"].bboxes).cuda()

        # 整形
        img_letterboxed = letterbox(image = img)
        # tensorに直す
        img_tensor = torch.from_numpy(img_letterboxed).permute(2, 0, 1).unsqueeze(0)
        img_tensor /= 255

        #推論
        result = model(img_tensor)[0].boxes
        bbox_pre = result.xyxy
        conf_pre = result.conf.tolist()
        cls_pre = result.cls.tolist()
        n = 0
        for gt in gt_bbox:
            if (bbox_pre.nelement()  == 0):
                 IoU_max, conf, clas = 0, 0, 0
            else:
                IoU = torchvision.ops.box_iou(bbox_pre, gt.unsqueeze(dim = 0))
                IoU_max, max_index = IoU.max().tolist(), IoU.argmax()
                conf = conf_pre[max_index]
                clas = cls_pre[max_index]
            
            if((clas == 20) & (IoU_max > 0.1)):
                anninfo["conf"].append(conf)
                anninfo["IoU"].append(IoU_max)
            else:
                anninfo["conf"].append(0)
                anninfo["IoU"].append(0)
            anninfo["ann_path"].append(ann_path+f" -{n}")
            n += 1
        anninfo["edit"] += edit

            

import json

with open("african_wildlife_elephant_pre.json", "w") as f: 
    json.dump(anninfo, f)