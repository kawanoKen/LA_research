import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from ultralytics import YOLO
import torch    
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
import torchvision.transforms as transforms
import torch
from ultralytics.data.augment import LetterBox
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# COLORS = np.random.uniform(0, 255, size=(80, 3))

# def parse_detections(results):
#     detections = results.pandas().xyxy[0]
#     detections = detections.to_dict()
#     boxes, colors, names = [], [], []

#     for i in range(len(detections["xmin"])):
#         confidence = detections["confidence"][i]
#         if confidence < 0.2:
#             continue
#         xmin = int(detections["xmin"][i])
#         ymin = int(detections["ymin"][i])
#         xmax = int(detections["xmax"][i])
#         ymax = int(detections["ymax"][i])
#         name = detections["name"][i]
#         category = int(detections["class"][i])
#         color = COLORS[category]

#         boxes.append((xmin, ymin, xmax, ymax))
#         colors.append(color)
#         names.append(name)
#     return boxes, colors, names


# def draw_detections(boxes, colors, names, img):
#     for box, color, name in zip(boxes, colors, names):
#         xmin, ymin, xmax, ymax = box
#         cv2.rectangle(
#             img,
#             (xmin, ymin),
#             (xmax, ymax),
#             color, 
#             2)

#         cv2.putText(img, name, (xmin, ymin - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
#                     lineType=cv2.LINE_AA)
#     return img


# image_path = "/work/kawano/LA/yolo_research/traffic.jpg"
# img = np.array(Image.open(image_path))
# img = cv2.resize(img, (640, 640))
# rgb_img = img.copy()
# img = np.float32(img) / 255
# transform = transforms.ToTensor()
# tensor = transform(img).unsqueeze(0)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model.eval()
# model.cpu()
# target_layers = [model.model.model.model[-2]]

# results = model([rgb_img])
# boxes, colors, names = parse_detections(results)
# detections = draw_detections(boxes, colors, names, rgb_img.copy())
# result = Image.fromarray(detections)
# result.save("/work/kawano/LA/yolo_research/traffic_predict.jpg")

# cam = EigenCAM(model, target_layers)
# grayscale_cam = cam(tensor)[0, :, :]
# cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
# result = Image.fromarray(cam_image)
# result.save("/work/kawano/LA/yolo_research/traffic_gradcam.jpg")
# breakpoint()

model = YOLO('yolov8n.pt')  # or yolov8s.pt, etc.
model.eval()


target_layers = [model.model.model[-2]]
img_path = "/work/kawano/LA/datasets/coco/images/val2017/000000000139.jpg"
img = cv2.imread(img_path)
img = img.astype(np.float32)
letterbox = LetterBox(auto = True)
img_let = letterbox(image = img)
img_tensor = torch.from_numpy(img_let).permute(2, 0, 1).unsqueeze(0)
img_tensor /= 255

outputs = model(img_tensor)
target_class = outputs[0].boxes.cls
targets = [ClassifierOutputTarget(category) for category in target_class]


cam = EigenCAM(model, target_layers)
result = cam(img_tensor,targets = targets)
cam_image = show_cam_on_image(img_let, result[0], use_rgb=True)

plt.imshow(cam_image)
plt.axis('off')
plt.title("Class Activation Map (CAM)")
plt.show()
plt.savefig("cam_output.png")