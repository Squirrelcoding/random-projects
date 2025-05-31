import os
import json
from typing import Any

import torch
from torch.utils.data import DataLoader

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional import to_dtype

import PIL.Image

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2
NUM_WORKERS = 2

class AirportDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, annotation_file: str, transform=None):
        self.transform = transform
        self.data = []

        with open(annotation_file) as f:
            annotations: dict = json.load(f)
            for entry in annotations.values():
                filename = entry["filename"]
                regions = entry["regions"]

                boxes = []
                for region in regions:
                    x = region["shape_attributes"]["x"]
                    y = region["shape_attributes"]["y"]
                    w = region["shape_attributes"]["width"]
                    h = region["shape_attributes"]["height"]
                    boxes.append([x, y, x + w, y + h])

                self.data.append({
                    "img_path": os.path.join(root_dir, filename),
                    "boxes": boxes
                })

    def __getitem__(self, idx: int) -> Any:
        entry = self.data[idx]
        img = PIL.Image.open(entry["img_path"]).convert("RGB")

        # Resize the image to 400 x 400 pixels and calculate the scale factors
        scale_x = 400 / img.width
        scale_y = 400 / img.height
        img = img.resize((400, 400))

        boxes = torch.tensor(entry["boxes"], dtype=torch.float32)

        # Scale the boxes
        boxes[:, 2] *= scale_x
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y

        # convert the image
        img = to_dtype(tv_tensors.Image(img), torch.float32)

        target = {}

        # Update boxes so that its a set of bounding boxes
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, 
            format="XYXY", 
            canvas_size=F.get_size(img) # type: ignore
        )
        target["labels"] = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target["image_id"] = torch.tensor(idx)
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

def collate_fn(batch):
    return tuple(zip(*batch))

dataset = AirportDataset("data/airports", "via_export_json.json")

# Form new training and testing dataloaders
N = len(dataset)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [N - int(0.1 * N), int(0.1 * N)])

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS # type: ignore
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS # type: ignore
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the head with a new one for 2 classes (background and airport)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
model.to(device)

model.train()

train_loss, train_acc = 0, 0

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, 
                                                   weight_decay=0.0005)

epochs = 5

for epoch in range(epochs):
    model.train()
    for images, targets in train_dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        pred = model(images, targets)
        losses = sum(loss for loss in pred.values())

        losses.backward()
        optimizer.step()
        train_loss += losses.item()

    print(f"Epoch {epoch + 1} / {epochs} done!")

model.eval()

with torch.no_grad():
    for images, targets in test_dataloader:
        images = list(img.to(device) for img in images)
        predictions = model(images)
        # Example: print the bounding boxes and labels for the first image
        print(predictions[0]['boxes'])
        print(predictions[0]['labels'])

import cv2
from PIL import Image
import torchvision.transforms.v2 as T


# Load image
img = Image.open("drive/MyDrive/data/airports/mpls.png").convert("RGB")
# Apply the same transformation as for training

transform = T.Compose([
    T.Resize((400, 400)),
    T.ToTensor(),
])

img = transform(img)
img = img.squeeze(0).to(device)
# print(img.shape)
img.permute(1, 2, 0)
# Model prediction
model.eval()
with torch.no_grad():
    prediction = model([img])
# Print the predicted bounding boxes and labels
print(prediction[0]['boxes'])
print(prediction[0]['labels'])