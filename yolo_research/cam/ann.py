from ultralytics import YOLO
import torch    
import cv2
import numpy as np
from ultralytics.utils.instance import Instances, Bboxes
from ultralytics.data.augment import LetterBox
import os
import glob

def yolo_to_xyxy(bbox, img_w, img_h):
    xc, yc, w, h = bbox
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]




model = YOLO('yolov8n.pt')  # or yolov8s.pt, etc.d
hook_layer = [model.model.model[i] for i in range(10)] + [model.model.model[i] for i in [12, 15, 18, 21]]
model(torch.zeros(1, 3, 640, 640))  # or yolov8s.pt, etc.d

conv_outputs = []
cnn_w = []

# フック関数定義
def get_activation(name):
    def hook(model, input, output):
            conv_outputs.append(output[0].cpu().detach().numpy().copy())
    return hook

# 全ての畳み込み層にフック登録
for idx, layer in enumerate(hook_layer):
    # cnn_w.append(layer.conv.weight[:, 0, :,:])
    layer.register_forward_hook(get_activation(f"{idx}-{layer.__class__.__name__}"))


important_layer = [[5, 2, 7, 15], 
                   [29, 6, 18, 20, 25, 26, 11, 16], 
                   [18, 5, 9, 6, 26, 27, 4, 13], 
                   [9, 45, 48, 5, 20, 41, 33, 58, 53, 57, 31, 32, 14, 15, 25, 49], 
                   [13, 53, 60, 39, 29, 15, 31, 48, 27, 59, 50, 56, 44, 9, 2, 57], 
                   [43, 114, 125, 112, 91, 123, 6, 4, 107, 104, 21, 110, 56, 45, 86, 57, 68, 126, 34, 92, 19, 50, 42, 7, 120, 108, 32, 26, 55, 83, 28, 95], 
                   [35, 97, 37, 82, 49, 80, 89, 105, 65, 96, 1, 7, 95, 32, 23, 98, 123, 70, 25, 90, 107, 34, 117, 19, 119, 2, 0, 84, 101, 24, 8, 112], 
                   [29, 52, 255, 67, 96, 41, 122, 191, 20, 153, 211, 93, 90, 202, 11, 33, 59, 107, 141, 45, 55, 54, 118, 65, 170, 13, 101, 208, 88, 149, 233, 75, 168, 174, 181, 239, 102, 70, 28, 225, 230, 248, 82, 69, 83, 56, 196, 243, 123, 254, 232, 235, 60, 48, 86, 113, 115, 205, 64, 121, 92, 188, 42, 229], 
                   [216, 223, 54, 130, 192, 200, 72, 44, 122, 70, 118, 49, 249, 163, 160, 74, 68, 166, 76, 215, 4, 218, 11, 195, 227, 209, 230, 60, 51, 244, 159, 168, 101, 252, 32, 207, 175, 52, 186, 97, 158, 233, 231, 31, 67, 95, 187, 92, 169, 84, 7, 102, 104, 38, 177, 96, 58, 85, 30, 23, 103, 37, 194, 57], 
                   [248, 254, 88, 191, 122, 100, 43, 246, 228, 144, 107, 50, 106, 203, 12, 235, 30, 195, 162, 205, 71, 125, 178, 7, 84, 111, 108, 131, 91, 0, 160, 15, 127, 209, 241, 130, 65, 35, 80, 109, 139, 207, 64, 128, 210, 152, 242, 252, 177, 166, 134, 213, 170, 116, 105, 169, 113, 19, 176, 237, 79, 114, 147, 67], 
                   [28, 95, 1, 22, 123, 51, 72, 24, 58, 59, 53, 2, 32, 37, 33, 91, 120, 103, 108, 14, 19, 96, 119, 45, 70, 99, 49, 11, 62, 65, 82, 117], 
                   [36, 53, 2, 20, 23, 63, 56, 35, 39, 34, 30, 5, 60, 37, 18, 12], 
                   [83, 57, 39, 60, 67, 99, 100, 86, 5, 92, 31, 104, 108, 7, 41, 103, 11, 85, 48, 36, 69, 107, 3, 79, 37, 19, 115, 94, 125, 97, 61, 2], 
                   [254, 122, 2, 36, 63, 0, 192, 170, 115, 95, 204, 10, 226, 14, 245, 18, 40, 91, 88, 246, 103, 169, 52, 19, 253, 126, 230, 72, 41, 235, 134, 135, 46, 124, 213, 153, 97, 102, 30, 196, 244, 248, 31, 156, 17, 29, 247, 62, 53, 116, 201, 163, 73, 99, 179, 12, 185, 108, 220, 13, 119, 186, 139, 83]]

layer_using = []
for layer in important_layer:
    layer_using.append([layer[0]])

img_exts=('.jpg', '.jpeg', '.png')
img_dir = "/work/kawano/LA/datasets/african-wildlife_noise/train/images"
ann_dir = "/work/kawano/LA/datasets/african-wildlife_noise/train/labels"
anninfo = {}
anninfo["ann_path"] = []
anninfo["ann_num"] = []
anninfo["max_feature_using"] = []
anninfo["max_feature_all"] = []
anninfo["mean_feature_using"] = []
anninfo["edit"] = []
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
            class_ids.append(class_id)
            edit.append(line[5])
        instance = Instances(bboxes=np.array(bboxes), segments = np.zeros((2,2)), bbox_format='xywh', normalized=True)
        instance.classes = np.array(class_ids)
        instance._bboxes.convert("xyxy")
        # label_dict = {"img": img, "instances": instance}
        # letterbox = LetterBox(auto = True)

        # letterboxed = letterbox(labels=label_dict)
        # img = letterboxed["img"]
        # gt_bbox = letterboxed["instances"].bboxes
        h, w = img.shape[0], img.shape[1]

        # アノテーション領域抽出
        imgs_ann = []
        box = instance.bboxes
        for i in range(len(box)):
            if (int(h*box[i][1]) == int(h*box[i][3])) or (int(h*box[i][0]) == int(h*box[i][2])):
                edit.pop(i)
                continue
            imgs_ann.append(img[int(h*box[i][1]):int(h*box[i][3]), int(w*box[i][0]):int(w*box[i][2]), :])


        imgs_tensor = []
        # 整形
        anninfo["edit"] += edit
        n=0
        for i in imgs_ann:
            img_letterboxed = letterbox(image = i)

        # tensorに直す
            img_tensor = torch.from_numpy(img_letterboxed).permute(2, 0, 1).unsqueeze(0)
            img_tensor /= 255
            img_tensor.requires_grad = True

        # 推論
            model(img_tensor)
            mean_feature_using = []
            max_feature_using = []
            max_feature_all = []
            for j in range(len(conv_outputs)):
                max_feature_using += [conv_outputs[j][k].max().tolist() for k in layer_using[j]]
                max_feature_all += [output.max().tolist() for output in conv_outputs[j]]
                mean_feature_using += [conv_outputs[j][k].mean().tolist() for k in layer_using[j]]
                 
            features.append(conv_outputs)
            anninfo["max_feature_all"].append(max_feature_all)
            anninfo["max_feature_using"].append(max_feature_using)
            anninfo["mean_feature_using"].append(mean_feature_using)
            anninfo["ann_path"].append(ann_path+f" -{n}")
            n += 1
            conv_outputs=[]
            

import json

with open("african_wildlife_elephant_toplayer.json", "w") as f: 
    json.dump(anninfo, f)
                 
