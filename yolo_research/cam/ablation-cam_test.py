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

            box = torch.tensor(self.bounding_boxes[None, 1])
            label = self.labels[1]
            if torch.cuda.is_available():
                box = box.cuda()
            ious = torchvision.ops.box_iou(box, output_bbox)[0]
            indexes = (ious > 0.8) & (output_cls.argmax(1) == label)
            #indexes = ious.argmax()
            score = (ious[indexes] + output_cls.max(1).values[indexes]).sum()
            target = target + score
            return target
            
            # for box, label in zip(self.bounding_boxes, self.labels):
            #     box = torch.Tensor(box[None, :])
            #     if torch.cuda.is_available():
            #         box = box.cuda()
            #     ious = torchvision.ops.box_iou(box, output_bbox)[0]
            #     indexes = (ious > 0.8) & (output_cls.argmax(1) == label)
            #     score = (ious[indexes] + output_cls.argmax(1)[indexes]).sum()
            #     target = target + score
            # return target
    
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

target_layers = [model.model.model[15], model.model.model[18], model.model.model[21]]
img_path = "/work/kawano/LA/datasets/coco/images/val2017/000000000139.jpg"
img = cv2.imread(img_path)
img = img.astype(np.float32)

ann_path = "/work/kawano/LA/yolo_research/coco_converted/labels/val2017/000000000139.txt"
bboxes = []
class_ids = []
segments = []
for line in open(ann_path, 'r'):
    line = list(map(float, line.strip().split()))
    class_id = line[0]
    bbox = line[1:5]
    segment = line[5:]
    bboxes.append(bbox)
    class_ids.append(class_id)
    segments.append(segment)
instance = Instances(bboxes=np.array(bboxes), segments = np.zeros((2,2)), bbox_format='xywh', normalized=True)
instance.classes = np.array(class_ids)
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

for grad in grads:
    b, c, w, h = grad.shape[0], grad.shape[1], grad.shape[2], grad.shape[3]
    max_grad = grad.view(b, c, w * h).max(2)[0]
    breakpoint()
cam_per_layer = cam.compute_cam_per_layer(img_tensor, target, False)

breakpoint()
plt.imshow(visualization)
plt.axis('off')
plt.savefig("grad-cam_output.png")
plt.show()