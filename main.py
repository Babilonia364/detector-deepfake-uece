import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import os
from resnet_classifier import get_resnet18
from evaluate import evaluate

device = torch.device('cuda') # or preffered device eg. cpu

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

data_dir = "Face storage location"
train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
test_data = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = get_resnet18()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to('cuda'), labels.to('cuda') #or preffered device
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    auc, acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1} Loss {running_loss:.2f} AUC {auc:.4f} Acc {acc:.4f}")
