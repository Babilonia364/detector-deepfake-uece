import torch
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def evaluate(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    auc_score = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, np.array(all_probs) > 0.5)
    print(f"Accuracy: {acc:.4f} AUC: {auc_score:.4f}")
    plot_auc_roc(all_labels, all_probs)
    return auc_score, acc
