import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def draw_yolo_with_matplotlib(image_path, label_path, class_names=None, save_path=None):
    # OpenCVで画像を読み込み、RGBに変換（Matplotlib表示用）
    image = cv2.imread(image_path)
    if image is None:
        print(f"画像が読み込めませんでした: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    # 描画準備
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # アノテーション読み込み
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            breakpoint()
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])

            # ピクセルに変換
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h

            # 左上座標
            x1 = x_center - width / 2
            y1 = y_center - height / 2

            # 矩形とラベル描画
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

            label = class_names[class_id] if class_names else str(class_id)
            ax.text(x1, y1 - 5, label, color='white', fontsize=10,
                    bbox=dict(facecolor='lime', alpha=0.5, boxstyle='round,pad=0.2'))

    ax.axis('off')  # 軸非表示

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"保存しました: {save_path}")
        plt.close()
    else:
        plt.show()

# === 使用例 ===
img_path = '/work/kawano/LA/datasets/african-wildlife_noise/train/images/2 (121).jpg'
label_path = '/work/kawano/LA/datasets/african-wildlife_noise/train/labels/2 (121).txt'
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

draw_yolo_with_matplotlib(img_path, label_path, save_path='/work/kawano/LA/yolo_research/2 (121).jpg')

