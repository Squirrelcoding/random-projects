import os
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from airport_dataset import AirportDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

torch.manual_seed(42)

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

data_transform = transforms.Compose([
    transforms.ToTensor()
])    


dataset = AirportDataset("data", "via_export_json.json", transform=data_transform)

# Form new training and testing dataloaders
N = len(dataset)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [N - int(0.1 * N), int(0.1 * N)])

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS # type: ignore
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS # type: ignore
)

model = fasterrcnn_resnet50_fpn(pretrained=True)

# 1. Get a batch of images and labels from the DataLoader
img_batch, label_batch = next(iter(train_dataloader))

img = img_batch[20]

# # 3. Perform a forward pass on a single image
model.eval()
with torch.inference_mode():
    pred = model(img_batch[0].unsqueeze(dim=0))
    boxes = pred[0]['boxes']
    scores = pred[0]['scores']
    labels = pred[0]['labels']


fig, ax = plt.subplots(1)
ax.imshow(img.permute(1, 2, 0))

# Add boxes
for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box.tolist()
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1 - 5, f'{label.item()} ({score:.2f})', color='red', fontsize=8, backgroundcolor='white')

plt.axis('off')
plt.savefig("img.png")
plt.show()