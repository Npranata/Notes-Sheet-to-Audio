import json
import shutil

import cv2
import os
from sklearn.model_selection import train_test_split

"""
Calculates the center of the bounding box's pixel coordinates we get from the annotation file and normalizes 
them to get it ready to be used for YOLO format. YOLO format is represented as [category id, x center ,y center,width,height].
"""
def convertBoundingBox(a_bbox, imgWidth, imgHeight):
    x_min, y_min, x_max, y_max = a_bbox
    x_center = ((x_min + x_max) / 2 ) / imgWidth
    y_center = ((y_min + y_max) / 2) / imgHeight
    width = (x_max - x_min ) / imgWidth
    height = (y_max - y_min ) / imgHeight
    return x_center, y_center, width, height

"""
Converts all bounding box annotations from the DeepScores JSON dataset into YOLO-compatible
label files and stores them in a new directory.

For each image in the dataset:
- Retrieve image's width, height, and category ID.
- Find all annotations associated with that image.
- Convert each bounding box from pixel coordinates to normalized YOLO format 
- Writes the converted annotations into a text file with the same name as the image.
"""
def convertJsonToYolo():
    with open("Dataset/ds2_dense/deepscores_train.json", "r") as f:
        data = json.load(f)
    label_dir = "Dataset/yolo_labels"
    os.makedirs(label_dir, exist_ok=True)

    annotations = data['annotations'] # Stores labels for images or other data to train models, often in formats like JSON
    imgs = data['images']

    for img in imgs:
        imgWidth = img['width']
        imgHeight = img['height']
        imgId = str(img['id']) # Get the image ID
        imgName = os.path.splitext(img["filename"])[0]
        print(imgName)

        # print(f"Processing {imgName} (ID: {imgId})")

        label_path = os.path.join(label_dir, f"{imgName}.txt")
        annotation_count = 0

        # Find all annotations for this image
        with open(label_path, "w") as yoloLabels:
            for ann_id, annotation in annotations.items():
                # Check if this annotation belongs to this image
                if str(annotation['img_id'] )== imgId:
                    bbox = annotation['a_bbox']
                    # print(f"The BBOX: {bbox}")
                    categoryId = annotation['cat_id']


                    # Handle categoryId being a list
                    if categoryId is None or (isinstance(categoryId, list) and None in categoryId):
                        continue
                    if isinstance(categoryId, list):
                        categoryId = int(categoryId[0]) - 1 #YOLO uses 0-based indexing

                    # Convert a_bbox(bounding box) to YOLO format
                    x_center, y_center, bboxWidth, bboxHeight = convertBoundingBox(bbox, imgWidth, imgHeight)
                    label_line = f"{categoryId} {x_center} {y_center} {bboxWidth} {bboxHeight}\n"
                    yoloLabels.write(label_line)
                    annotation_count += 1

        print(f"Saved labels to {label_path}")


def splitData():
    img_dir = "Dataset/ds2_dense/images"
    yoloLabel_dir = "Dataset/yolo_labels"
    for newDir in ["train", "val"]:
        os.makedirs(f"Dataset/images/{newDir}", exist_ok=True)
        os.makedirs(f"Dataset/labels_filtered/{newDir}", exist_ok=True)
    yoloFiles = [y for y in os.listdir(yoloLabel_dir) if y.endswith(".txt")]
    trainFiles, valFiles = train_test_split(yoloFiles, train_size=0.8, random_state=42)
    def copyFiles(files, subsetFile):
        for file in files:
            # Copy yolo label files into a new folder that splits them into training and validation files.
            label_src = os.path.join(yoloLabel_dir, file)
            label_dst = os.path.join(f"Dataset/labels_filtered/{subsetFile}", file)
            shutil.copy(label_src, label_dst)

            #Copy Image files into a new folder that splits them into training and validation files.
            img_name = os.path.splitext(file)[0]
            img_src = os.path.join(img_dir, img_name + ".png")
            if os.path.exists(img_src):
                img_dst = os.path.join(f"Dataset/images/{subsetFile}", img_name + ".png")
                shutil.copy(img_src, img_dst)
    copyFiles(trainFiles, "train")
    copyFiles(valFiles, "val")


#Convert and split data
# convertJsonToYolo()
# splitData()


# import os
# import yaml

# # === CONFIG ===
# LABELS_DIR = "Dataset/yolo_labels"     # folder where YOLO .txt files are now
# OUTPUT_DIR = "Dataset/yolo_labels_filtered"  # folder to save filtered labels
# DATASET_YAML = "Dataset/dataset.yaml"          # your dataset.yaml path

# # === STEP 1: Load class names from dataset.yaml ===
# with open(DATASET_YAML, "r") as f:
#     data = yaml.safe_load(f)
# all_classes = data["names"]

# # === STEP 2: Choose which classes to keep ===
# KEEP_CLASSES = [
#     "clefG", "clefF", "clefCAlto", "clefCTenor",
#     "noteheadBlackOnLine", "noteheadBlackInSpace",
#     "noteheadHalfOnLine", "noteheadHalfInSpace",
#     "noteheadWholeOnLine", "noteheadWholeInSpace",
#     "ledgerLine",
#     "stem", "flag8thUp", "flag8thDown", "flag16thUp", "flag16thDown",
#     "flag32ndUp", "flag32ndDown", "augmentationDot", "tie", "beam",
#     "restWhole", "restHalf", "restQuarter", "rest8th",
#     "rest16th", "rest32nd", "rest64th",
#     "accidentalSharp", "accidentalFlat", "accidentalNatural",
#     "keySharp", "keyFlat", "keyNatural",
#     "staff", "brace",
#     "timeSig0", "timeSig1", "timeSig2", "timeSig3", "timeSig4",
#     "timeSig5", "timeSig6", "timeSig7", "timeSig8", "timeSig9",
#     "timeSigCommon", "timeSigCutCommon",
#     "slur"
# ]

# # === STEP 3: Find their class IDs ===
# keep_ids = {i for i, name in enumerate(all_classes) if name in KEEP_CLASSES}
# print(f"Keeping {len(keep_ids)} of {len(all_classes)} classes")

# # === STEP 4: Filter label files ===
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# kept, removed = 0, 0

# for fname in os.listdir(LABELS_DIR):
#     if not fname.endswith(".txt"):
#         continue

#     in_path = os.path.join(LABELS_DIR, fname)
#     out_path = os.path.join(OUTPUT_DIR, fname)

#     with open(in_path, "r") as fin:
#         lines = fin.readlines()

#     new_lines = []
#     for line in lines:
#         if not line.strip():
#             continue
#         class_id = int(line.split()[0])
#         if class_id in keep_ids:
#             new_lines.append(line)
#             kept += 1
#         else:
#             removed += 1

#     if new_lines:
#         with open(out_path, "w") as fout:
#             fout.writelines(new_lines)

# print(f"✅ Done. Kept {kept} boxes, removed {removed}. Filtered labels saved to {OUTPUT_DIR}")
