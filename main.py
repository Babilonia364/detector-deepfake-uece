import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from models.resnet_classifier import get_resnet18

# ====== Config ======
DEVICE = torch.device('cpu')

# ====== Transforms ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ====== Datasets & Loaders ======
train_ds = datasets.ImageFolder("frames/train", transform=transform)
test_ds = datasets.ImageFolder("frames/test", transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# ====== Model ======
model = get_resnet18().to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ====== Training Loop ======
def evaluate(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            prob = torch.softmax(out, dim=1)[:, 1]
            preds.extend(prob.cpu().numpy())
            labels.extend(y.cpu().numpy())
    auc = roc_auc_score(labels, preds)
    pred_labels = np.array(preds) > 0.5
    acc = accuracy_score(labels, pred_labels)
    return auc, acc

print("Starting Training")

for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")