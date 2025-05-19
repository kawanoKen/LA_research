import os
import json

def coco_json_to_yolo(json_path, output_dir, target_cls_ids):
    with open(json_path, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    cat_map = {cat['id']: cat['name'] for cat in coco['categories']}

    for cid in target_cls_ids:
        cname = cat_map.get(cid, str(cid))
        os.makedirs(os.path.join(output_dir, cname), exist_ok=True)

    for ann in coco['annotations']:
        cid = ann['category_id']
        if cid not in target_cls_ids:
            continue

        img = images[ann['image_id']]
        w, h = img['width'], img['height']
        x, y, bw, bh = ann['bbox']
        cx = (x + bw/2) / w
        cy = (y + bh/2) / h
        nw = bw / w
        nh = bh / h

        cname = cat_map[cid]
        out_dir = os.path.join(output_dir, cname)
        base, _ = os.path.splitext(img['file_name'])
        txt_path = os.path.join(out_dir, base + '.txt')

        with open(txt_path, 'a') as fout:
            fout.write(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    print("JSON→YOLO 変換＆クラス分割 完了")


def filter_existing_yolo(label_dir, output_dir, target_labels):
    for lbl in target_labels:
        os.makedirs(os.path.join(output_dir, str(lbl)), exist_ok=True)

    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        src = os.path.join(label_dir, fname)
        with open(src, 'r') as f:
            lines = f.readlines()

        for lbl in target_labels:
            out_lines = [l for l in lines if int(l.split()[0]) == lbl]
            if not out_lines:
                continue
            dst_dir = os.path.join(output_dir, str(lbl))
            with open(os.path.join(dst_dir, fname), 'w') as fout:
                fout.writelines(out_lines)

    print("既存YOLOラベルのフィルタ＆クラス分割 完了")


#
#
# category_IDをそのままclsにいているので要修正!!!!!!!!!!!!!!!
#
#



if __name__ == '__main__':
    # === ここを書き換えて使ってください ===
    MODE = 'from_json'      # 'from_json' or 'filter_yolo'
    JSON_PATH    = '/work/kawano/LA/datasets/coco/annotations/instances_val2017.json'
    LABEL_DIR    = '/work/kawano/LA/yolo_research/coco_converted/labels/val2017'
    OUTPUT_DIR   = '/work/kawano/LA/yolo_research/coco_test'
    CLASSES      = [22]   # 抽出したい COCO カテゴリ ID または YOLO クラス ID のリスト

    if MODE == 'from_json':
        coco_json_to_yolo(JSON_PATH, OUTPUT_DIR, CLASSES)
    elif MODE == 'filter_yolo':
        filter_existing_yolo(LABEL_DIR, OUTPUT_DIR, CLASSES)
    else:
        raise ValueError(f"Unknown MODE: {MODE}")