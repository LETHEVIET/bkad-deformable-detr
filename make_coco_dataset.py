import os
from tqdm import tqdm
import json
from PIL import Image
import shutil

data_root='/root/.cache/kagglehub/datasets/lngphmtrntrung/test-data-2/versions/1/'
json_file_path = f'{data_root}/data/split_train_val.json'
image_source_dir = f'{data_root}/data/images'
label_source_dir = f'{data_root}/data/labels'

train_image_dir = './dataset/train/images'
val_image_dir = './dataset/val/images'

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)

categories = [
    {"id": 0, "name": 'motorbike'},
    {"id": 1, "name": 'automobile'},
    {"id": 2, "name": 'passenger car'},
    {"id": 3, "name": 'truck'}
]

# Load the JSON file containing the filenames
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

splits = ['train', 'val']

for split in splits:
    print(split)
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }
    for image_file in data[split]:
        id = len(coco_dataset["images"])
        image_path = os.path.join(image_source_dir, image_file)
        shutil.copy(image_path, f'./dataset/{split}/images')
        image = Image.open(image_path)
        width, height = image.size

        # Add the image to the COCO dataset
        image_dict = {
            "id": id,
            "width": width,
            "height": height,
            "file_name": 'images/' + image_file
        }

        coco_dataset["images"].append(image_dict)

        if image_file == 'daytime_cam_10_00500.jpg':
            image_file = 'daytime_cam_10_000500.jpg'

        with open(os.path.join(label_source_dir, image_file[:-3] + 'txt')) as f:
            annotations = f.readlines()

        for ann in annotations:
            cls, x, y, w, h = map(float, ann.strip().split())
            cls = int(cls)
            x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
            ann_dict = {
                "id": len(coco_dataset["annotations"]),
                "image_id": id,
                "category_id": cls,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(ann_dict)

    # Save the COCO dataset to a JSON file
    with open(f'./dataset/{split}/annotations.json', 'w') as f:
        json.dump(coco_dataset, f)


print("COCO format Dataset was created successfully.")