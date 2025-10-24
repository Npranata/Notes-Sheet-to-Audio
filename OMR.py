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
                        categoryId = categoryId[0]

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
        os.makedirs(f"Dataset/split_images/{newDir}", exist_ok=True)
        os.makedirs(f"Dataset/split_yolo_labels/{newDir}", exist_ok=True)
    yoloFiles = [y for y in os.listdir(yoloLabel_dir) if y.endswith(".txt")]
    trainFiles, valFiles = train_test_split(yoloFiles, train_size=0.8, random_state=42)
    def copyFiles(files, subsetFile):
        for file in files:
            # Copy yolo label files into a new folder that splits them into training and validation files.
            label_src = os.path.join(yoloLabel_dir, file)
            label_dst = os.path.join(f"Dataset/split_yolo_labels/{subsetFile}", file)
            shutil.copy(label_src, label_dst)

            #Copy Image files into a new folder that splits them into training and validation files.
            img_name = os.path.splitext(file)[0]
            img_src = os.path.join(img_dir, img_name + ".png")
            if os.path.exists(img_src):
                img_dst = os.path.join(f"Dataset/split_images/{subsetFile}", img_name + ".png")
                shutil.copy(img_src, img_dst)
    copyFiles(trainFiles, "train")
    copyFiles(valFiles, "val")


#Convert and split data
convertJsonToYolo()
splitData()


