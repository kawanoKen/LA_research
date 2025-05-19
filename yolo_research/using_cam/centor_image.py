#!/usr/bin/env python3
# bbox_center_matplotlib_norm.py
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple


def yolo2pixel_bbox(
    xc_n: float, yc_n: float, w_n: float, h_n: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """
    正規化 YOLO BBox → 画素座標 (x, y, w, h)
    """
    w_px, h_px = w_n * img_w, h_n * img_h
    x_px = xc_n * img_w - w_px / 2
    y_px = yc_n * img_h - h_px / 2

    # 範囲クリップ & 整数化
    x1 = max(int(round(x_px)), 0)
    y1 = max(int(round(y_px)), 0)
    x2 = min(int(round(x_px + w_px)), img_w)
    y2 = min(int(round(y_px + h_px)), img_h)
    return x1, y1, x2 - x1, y2 - y1


def bbox_center_pseudo_norm(
    img_path: str | Path,
    bbox_norm: Tuple[float, float, float, float],
    *,
    debug: bool = True,
    save_path: str | Path | None = "debug_center_norm.png",
) -> Tuple[float, float]:
    """
    正規化 BBox (xc, yc, w, h) → 擬似中心 (cx, cy) [pixel]
    """
    # 1) 画像読み込み
    img = mpimg.imread(str(img_path))
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    H, W = img.shape[:2]

    # 2) 正規化 bbox → pixel bbox
    x, y, w, h = yolo2pixel_bbox(*bbox_norm, img_w=W, img_h=H)

    # 3) クロップ
    crop = img[y : y + h, x : x + w]

    # 4) グレースケール
    if crop.ndim == 3:
        r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
        crop_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        crop_gray = crop
    breakpoint()
    # 5) 二値化 (平均輝度閾値)
    thresh = crop_gray.mean()
    mask = crop_gray > thresh
    if not mask.any():  # 真っ黒 ⇒ 幾何中心
        return x + w / 2, y + h / 2

    # 6) 重心
    ys, xs = np.nonzero(mask)
    cx_local, cy_local = xs.mean(), ys.mean()
    cx, cy = x + cx_local, y + cy_local

    # 7) デバッグ可視化
    if debug:
        fig, ax = plt.subplots()
        ax.imshow(img)
        rect = plt.Rectangle((x, y), w, h, fill=False, lw=2, color="red")
        ax.add_patch(rect)
        ax.scatter([cx], [cy], s=40, marker="x", color="cyan")
        ax.set_title(f"pseudo center = ({cx:.1f}, {cy:.1f})")
        ax.axis("off")
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[INFO] saved debug image → {save_path}")
        else:
            plt.show()
        plt.close(fig)

    return cx, cy


# ---------- サンプル実行 ----------
if __name__ == "__main__":
    IMG = "/work/kawano/LA/datasets/african-wildlife/train/images/2 (340).jpg"              # 画像パス
    BBOX_NORM = (0.675781, 0.387838, 0.345312, 0.583784)  # xc, yc, w, h (0–1)
    cx, cy = bbox_center_pseudo_norm(IMG, BBOX_NORM)
    print(f"Pseudo center: ({cx:.1f}, {cy:.1f})")