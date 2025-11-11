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
- Early stopping e ajuste de learning rate
- Verificação de balanceamento de classes

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import os
from resnet_classifier import get_model
from evaluate import evaluate
from tqdm import tqdm
from focal_loss import FocalLoss

# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

train_transfor = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # crop aleatório
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
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

# Carregar dados
train_data = datasets.ImageFolder(os.path.join(data_dir, "trained"), transform=train_transfor)
test_data = datasets.ImageFolder(os.path.join(data_dir, "tested"), transform=val_transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Verificar balanceamento de classes
train_labels = [label for _, label in train_data]
class_counts = np.bincount(train_labels)
print(f"Distribuição de classes no treino: {class_counts}")
print(f"Proporção: {class_counts[0]/sum(class_counts):.3f} vs {class_counts[1]/sum(class_counts):.3f}")

# Se as classes estiverem desbalanceadas, adicionar peso
if abs(class_counts[0] - class_counts[1]) > 0.2 * sum(class_counts):
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights.to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    alpha = torch.tensor([0.4, 0.8], device=device)
    criterion = FocalLoss(alpha=alpha, gamma=2.5)
    print("Usando pesos para balanceamento de classes")
else:
    criterion = nn.CrossEntropyLoss()
    print("Classes balanceadas, usando CrossEntropyLoss padrão")

# Modelo e otimizador
model = get_model()
model = model.to(device)

# Otimizador com weight decay para regularização
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # LR menor + regularização

# Scheduler para ajustar learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Early stopping
best_f1 = 0
best_epoch = 0
patience = 5
epochs_without_improvement = 0
total_batches = len(train_loader)

print(f"\nIniciando treinamento por até 20 épocas...")
print(f"Tamanho do dataset de treino: {len(train_data)}")
print(f"Tamanho do dataset de teste: {len(test_data)}")

for epoch in range(20):  # Aumentado para 20 épocas máximo
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/20")
    print(f"Learning rate atual: {scheduler.get_last_lr()[0]:.2e}")
    
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    with tqdm(train_loader, unit="batch", desc=f"Treinando Epoch {epoch+1}") as pbar:
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calcular acurácia durante treinamento
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Atualizar barra de progresso
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
            })

    # Avaliar no conjunto de teste
    auc, acc, f1, recall, precision = evaluate(model, test_loader, device)
    avg_loss = running_loss / total_batches
    
    print(f"Resultados - Loss: {running_loss:.4f} | AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    
    # Early stopping logic
    if f1 > best_f1:
        best_f1 = f1
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        # Salvar melhor modelo
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'f1': f1,
            'auc': auc,
            'acc': acc,
            'precision': precision,
            'recall': recall
        }, 'best_model.pth')
        print(f"Novo melhor modelo salvo! F1: {f1:.4f}")
    else:
        epochs_without_improvement += 1
        print(f"Sem melhoria ({epochs_without_improvement}/{patience})")
    
    # Atualizar learning rate
    scheduler.step()
    
    # Verificar early stopping
    if epochs_without_improvement >= patience:
        print(f"\nEarly stopping ativado na época {epoch+1}!")
        print(f"Melhor f1: {best_f1:.4f} na época {best_epoch}")
        break

print(f"\n{'='*50}")
print("Treinamento concluído!")
print(f"Melhor resultado: f1 {best_f1:.4f} na época {best_epoch}")

# Carregar melhor modelo para uso final
if os.path.exists('best_model.pth'):
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Melhor modelo carregado para avaliação final")
    
    # Avaliação final com melhor modelo
    final_auc, final_acc, final_f1, final_recall, final_precision = evaluate(model, test_loader, device)
    print(f"Resultado final - AUC: {final_auc:.4f} | Acc: {final_acc:.4f} | F1-score: {final_f1:.4f} | Precision: {final_precision:.4f} | Recall: {final_recall:.4f}")

    # ===============================================================
    # ANÁLISE COM GRAD-CAM (CHAMADA SIMPLES)
    # ===============================================================
    
    # Importar módulo Grad-CAM
    from grad_cam import analyze_predictions_with_cam
    
    print("\n" + "="*60)
    print("INICIANDO ANÁLISE COM GRAD-CAM")
    print("="*60)
    
    # Analisar 5 exemplos com Grad-CAM
    results = analyze_predictions_with_cam(
        model=model,
        test_data=test_data,
        device=device,
        num_examples=5,
        output_dir="gradcam_analysis"
    )
    
    print("\nAnálise com Grad-CAM concluída!")
    print("Os mapas de calor mostram as regiões que o modelo considera importantes para a decisão.")