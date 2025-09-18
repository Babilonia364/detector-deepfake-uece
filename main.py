"""
Deepfake Detection Trainer
==========================
Este script treina um classificador ResNet-18 para detectar deepfakes
usando faces extraídas de vídeos.

Funcionalidades:
- Carrega e pré-processa imagens de faces (reais e falsas)
- Treina um modelo ResNet-18 com transfer learning
- Avalia o modelo usando AUC e Accuracy
- Suporte para GPU/CPU

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import os
from resnet_classifier import get_resnet18
from evaluate import evaluate
from tqdm import tqdm

device = torch.device('cuda') # or preffered device eg. cpu

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

data_dir = "data"  # ← ALTERE PARA SEU CAMINHO REAL

# Verificar se os diretórios existem
train_path = os.path.join(data_dir, "trained")
test_path = os.path.join(data_dir, "tested")

if not os.path.exists(train_path):
    print(f"ERRO: Diretório de treino não encontrado: {train_path}")
    print("Execute primeiro o run_extraction.py para extrair as faces")
    exit()

if not os.path.exists(test_path):
    print(f"ERRO: Diretório de teste não encontrado: {test_path}")
    print("Certifique-se de ter dados de teste na estrutura correta")
    exit()

train_data = datasets.ImageFolder(os.path.join(data_dir, "trained"), transform=transform)
test_data = datasets.ImageFolder(os.path.join(data_dir, "tested"), transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = get_resnet18()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

total_batches = len(train_loader)

for epoch in range(10):
    model.train()
    running_loss = 0
    with tqdm(train_loader, unit="batch", desc=f"Treinando Epoch {epoch+1}") as pbar:
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device) #or preffered device
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Atualizar a barra de progresso com informações em tempo real
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
            })

    auc, acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1} Loss {running_loss:.2f} AUC {auc:.4f} Acc {acc:.4f}")
