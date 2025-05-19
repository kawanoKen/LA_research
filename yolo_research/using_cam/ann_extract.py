import os
import cv2

# パス設定
image_dir = "/work/kawano/LA/datasets/african-wildlife_noise/train/images"       # 元画像のディレクトリ
label_dir = "/work/kawano/LA/datasets/african-wildlife_noise/train/labels"       # YOLO形式アノテーションファイルのディレクトリ
output_dir = "/work/kawano/LA/yolo_research/dataset"     # 切り出し画像の保存先

os.makedirs(output_dir, exist_ok=True)

# 全画像を処理
for filename in os.listdir(label_dir):
    if not filename.endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    with open(label_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            class_id, x_center, y_center, bbox_w, bbox_h, _ = map(float, line.strip().split())

            # YOLOの正規化座標をピクセルに変換
            x1 = int((x_center - bbox_w / 2) * w)
            y1 = int((y_center - bbox_h / 2) * h)
            x2 = int((x_center + bbox_w / 2) * w)
            y2 = int((y_center + bbox_h / 2) * h)

            # 切り出し（画像の範囲外に出ないように制限）
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            cropped = img[y1:y2, x1:x2]

            # 保存
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{i}_class{int(class_id)}.jpg")
            cv2.imwrite(output_path, cropped)