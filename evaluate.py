import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, recall_score, precision_score
import numpy as np
import matplotlib.pyplot as plt

def plot_auc_roc(true_labels, predicted_probs):
    """Plota a curva ROC AUC"""
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calcular todas as métricas
    auc_score = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, np.array(all_probs) > 0.5)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    # print(f"Accuracy: {acc:.4f} AUC: {auc_score:.4f}")
    # plot_auc_roc(all_labels, all_probs)
    return auc_score, acc, f1, recall, precision
