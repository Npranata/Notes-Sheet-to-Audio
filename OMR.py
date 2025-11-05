import json, os, shutil, yaml
from sklearn.model_selection import train_test_split

# ========= CONFIG =========
JSON_PATH = "Dataset/ds2_dense/deepscores_train.json"
IMG_DIR = "Dataset/ds2_dense/images"
LABELS_DIR = "Dataset/yolo_labels"
FILTERED_DIR = "Dataset/yolo_labels_filtered"

TRAIN_DIR = "Dataset/train"
VAL_DIR = "Dataset/val"

# Important classes to keep
KEEP_CLASSES = [
    "clefG", "clefF", "clefCAlto", "clefCTenor",
    "noteheadBlackOnLine", "noteheadBlackInSpace",
    "noteheadHalfOnLine", "noteheadHalfInSpace",
    "noteheadWholeOnLine", "noteheadWholeInSpace",
    "ledgerLine",
    "stem", "flag8thUp", "flag8thDown", "flag16thUp", "flag16thDown",
    "flag32ndUp", "flag32ndDown", "augmentationDot", "tie", "beam",
    "restWhole", "restHalf", "restQuarter", "rest8th",
    "rest16th", "rest32nd", "rest64th",
    "accidentalSharp", "accidentalFlat", "accidentalNatural",
    "keySharp", "keyFlat", "keyNatural",
    "staff", "brace",
    "timeSig0", "timeSig1", "timeSig2", "timeSig3", "timeSig4",
    "timeSig5", "timeSig6", "timeSig7", "timeSig8", "timeSig9",
    "timeSigCommon", "timeSigCutCommon",
    "slur"
]


def convert_bbox(a_bbox, w, h):
    x_min, y_min, x_max, y_max = a_bbox
    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h
    return x_center, y_center, bw, bh



def convert_json_to_yolo():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    os.makedirs(LABELS_DIR, exist_ok=True)

    imgs = data["images"]
    annotations = data["annotations"]

    for img in imgs:
        img_id = str(img["id"])
        img_name = os.path.splitext(img["filename"])[0]
        label_path = os.path.join(LABELS_DIR, f"{img_name}.txt")
        img_w, img_h = img["width"], img["height"]

        with open(label_path, "w") as out:
            for _, ann in annotations.items():
                if str(ann["img_id"]) != img_id:
                    continue
                bbox = ann["a_bbox"]
                cat_id = ann["cat_id"] # category id
                if cat_id is None:
                    continue
                if isinstance(cat_id, list):
                    cat_id = cat_id[0]
                cat_id = int(cat_id) - 1  # ensure integer and make zero-based


                x, y, w, h = convert_bbox(bbox, img_w, img_h)
                out.write(f"{cat_id} {x} {y} {w} {h}\n")

    print("✅ Converted JSON → YOLO labels.")


# ========= Filter classes =========
def filter_labels():
    # Load original class names from dataset.yaml
    with open("Dataset/dataset.yaml", "r") as f:
        data = yaml.safe_load(f)
    all_classes = data["names"]

    keep_ids = {i for i, name in enumerate(all_classes) if name in KEEP_CLASSES}
    id_map = {old: new for new, old in enumerate(sorted(keep_ids))}

    os.makedirs(FILTERED_DIR, exist_ok=True)
    kept, removed = 0, 0

    for fname in os.listdir(LABELS_DIR):
        if not fname.endswith(".txt"):
            continue

        in_path = os.path.join(LABELS_DIR, fname)
        out_path = os.path.join(FILTERED_DIR, fname)

        new_lines = []
        for line in open(in_path):
            if not line.strip():
                continue
            cid = int(line.split()[0])
            if cid in keep_ids:
                new_cid = id_map[cid]
                new_lines.append(line.replace(str(cid), str(new_cid), 1))
                kept += 1
            else:
                removed += 1

        if new_lines:
            with open(out_path, "w") as f:
                f.writelines(new_lines)

    print(f"✅ Filtered: kept {kept}, removed {removed}")


def split_data():
    os.makedirs(TRAIN_DIR + "/images", exist_ok=True)
    os.makedirs(TRAIN_DIR + "/labels", exist_ok=True)
    os.makedirs(VAL_DIR + "/images", exist_ok=True)
    os.makedirs(VAL_DIR + "/labels", exist_ok=True)

    yolo_files = [f for f in os.listdir(FILTERED_DIR) if f.endswith(".txt")]
    train, val = train_test_split(yolo_files, train_size=0.8, random_state=42)

    def copy_split(files, subset):
        for f in files:
            img_name = os.path.splitext(f)[0] + ".png"
            img_src = os.path.join(IMG_DIR, img_name)
            label_src = os.path.join(FILTERED_DIR, f)

            if os.path.exists(img_src):
                shutil.copy(img_src, f"Dataset/{subset}/images/{img_name}")
            shutil.copy(label_src, f"Dataset/{subset}/labels/{f}")

    copy_split(train, "train")
    copy_split(val, "val")
    print("✅ Split into train/val folders.")


#  Write new dataset.yaml 
def write_new_yaml():
    new_yaml = {
        "path": "Dataset",
        "train": "train/images",
        "val": "val/images",
        "nc": len(KEEP_CLASSES),
        "names": KEEP_CLASSES
    }
    with open("Dataset/dataset_filtered.yaml", "w") as f:
        yaml.dump(new_yaml, f)
    print("✅ Wrote new dataset_filtered.yaml")


# ========= RUN ALL STEPS =========
# convert_json_to_yolo()
# filter_labels()
# split_data()
# write_new_yaml()

import cv2
import os
from matplotlib import pyplot as plt

# Update these paths
img_dir = "Dataset/train/images"
label_dir = "Dataset/train/labels"

# List all images
images = [f for f in os.listdir(img_dir) if f.endswith(('.png'))]

# Pick a random one to test
img_path = os.path.join(img_dir, images[0])
label_path = os.path.join(label_dir, images[0].replace('.png', '.txt').replace('.jpg', '.txt'))

# Load image
img = cv2.imread(img_path)
h, w, _ = img.shape

# Draw boxes
with open(label_path, "r") as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.strip().split())
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(int(cls)), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
