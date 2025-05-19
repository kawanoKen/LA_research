from ultralytics import YOLO
import os
import sys
project_name = "test"
sys.stdout = open(os.path.join("/work/kawano/LA/yolo_research/log", project_name + '.log'), 'w')
# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# バックボーンを凍結
# Train the model
#results = model.train(data="african-wildlife_noise.yaml", epochs=20, imgsz=640, verbose=False, )
results = model.train(data="african-wildlife_noise.yaml", project="/work/kawano/LA/ultralytics/runs/detect", name=project_name, epochs=20, imgsz=640, freeze=23, device = [0])
