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

      # Resize and scale boxes (same as before)
      scale_x = 400 / img.width
      scale_y = 400 / img.height
      img = img.resize((400, 400))

      boxes = torch.tensor(entry["boxes"], dtype=torch.float32)
      boxes[:, [0, 2]] *= scale_x  # x1, x2
      boxes[:, [1, 3]] *= scale_y  # y1, y2

      # Convert to tensor and normalize
      img = F.to_dtype(F.to_image(img), torch.float32, scale=True)  # [0,1] range
      img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

      target = {
          "boxes": boxes,
          "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),
          "image_id": torch.tensor(idx),
          "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
          "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
      }

      if self.transform:
          img, target = self.transform(img, target)

      return img, target

    def __len__(self) -> int:
        return len(self.data)

def collate_fn(batch):
    return tuple(zip(*batch))

dataset = AirportDataset("drive/MyDrive/data/airports", "drive/MyDrive/data/via_export_json.json")

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

epoch_loss, train_acc = 0, 0

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
        epoch_loss += losses.item()
    
    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader):.4f}")

import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T

# WRITTEN BY DEEPSEEK cause im too lazy

# 1. Pick a random image from the test dataset
random_idx = random.randint(0, len(test_dataset) - 1)
image, target = test_dataset[random_idx]  # Get image and ground truth (unused here)

# 2. Convert image tensor back to numpy for visualization
def denormalize(tensor):
    """Reverse normalization for display"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor.cpu() * std + mean

# Denormalize and permute dimensions (C, H, W) -> (H, W, C)
vis_img = denormalize(image).permute(1, 2, 0).numpy()
vis_img = np.clip(vis_img, 0, 1)  # Ensure pixel values are valid

# 3. Run inference
model.eval()
with torch.no_grad():
    prediction = model([image.to(device)])[0]  # Get first (and only) prediction

# 4. Filter predictions by confidence threshold
confidence_threshold = 0.5
keep = prediction['scores'] > confidence_threshold
boxes = prediction['boxes'][keep].cpu().numpy()
scores = prediction['scores'][keep].cpu().numpy()
labels = prediction['labels'][keep].cpu().numpy()

# 5. Plot the image with predicted boxes
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(vis_img)

# Add bounding boxes and labels
for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    # Draw rectangle
    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    # Add label (e.g., "Airport: 0.95")
    label_text = f"Airport: {score:.2f}"
    ax.text(
        x1, y1 - 5, label_text,
        color='red', fontsize=12, weight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

plt.axis('off')
plt.title(f"Predictions for Random Test Image (Index: {random_idx})")
plt.show()