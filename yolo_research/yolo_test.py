from ultralytics import YOLO
import pandas as pd

# Load a model
project_name = "world_life_noise_base(preprocessed)"
model = YOLO("/work/kawano/LA/ultralytics/runs/detect/" + project_name + "/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val(project="/work/kawano/LA/ultralytics/runs/detect", name="val_" + project_name)  # no arguments needed, dataset and settings remembered
cm      = metrics.confusion_matrix.matrix         # ← NumPy 配列 (nc+1, nc+1)
names   = list(model.names.values()) + ["background"]   # 行列の行・列ラベル

# DataFrame にするとわかりやすい
df = pd.DataFrame(cm.astype(int), index=names, columns=names)
print(df)