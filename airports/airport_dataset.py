import os
from typing import Any
import PIL.Image
import torch
import json

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

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # All class 1 (e.g. airport)

        # [x1, y1, x2, y2]
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(idx),
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
