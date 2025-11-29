import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from evaluate import evaluate
from resnet_classifier import get_model
import os

# Configurações
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = "data"

# Transformações (usar as mesmas do validation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def load_best_model(model_path='best_model.pth'):
    """Carrega o melhor modelo salvo"""
    model = get_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint

def get_all_predictions(model, loader, device):
    """Obtém todas as predições e labels"""
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
    
    return np.array(all_probs), np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(true_labels, predictions, class_names=['Real', 'Fake']):
    """Plota matriz de confusão"""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(true_labels, predicted_probs):
    """Plota curva ROC com AUC"""
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc

def plot_precision_recall_curve(true_labels, predicted_probs):
    """Plota curva Precision-Recall"""
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
    avg_precision = np.mean(precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_history(checkpoint):
    """Plota histórico de métricas (se disponível)"""
    # Esta função seria usada se você salvasse o histórico durante o treinamento
    print("Histórico de métricas não disponível no checkpoint atual")
    print("Para versão futura, salve as métricas por época durante o treinamento")

def plot_probability_distribution(true_labels, predicted_probs):
    """Plota distribuição das probabilidades preditas"""
    plt.figure(figsize=(10, 6))
    
    # Separar por classe verdadeira
    real_probs = predicted_probs[true_labels == 0]
    fake_probs = predicted_probs[true_labels == 1]
    
    plt.hist(real_probs, bins=50, alpha=0.7, label='Videos Reais', color='blue')
    plt.hist(fake_probs, bins=50, alpha=0.7, label='Deepfakes', color='red')
    plt.xlabel('Probabilidade Predita (Deepfake)')
    plt.ylabel('Frequência')
    plt.title('Distribuição das Probabilidades Preditas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Carregando modelo e dados...")
    
    # Carregar modelo
    model, checkpoint = load_best_model()
    
    # Carregar dados de teste
    test_data = datasets.ImageFolder(os.path.join(data_dir, "tested"), transform=val_transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    print(f"Modelo carregado - Melhor F1: {checkpoint['f1']:.4f}")
    print(f"Dataset de teste: {len(test_data)} amostras")
    
    # Obter todas as predições
    print("\nObtendo predições...")
    probs, true_labels, predictions = get_all_predictions(model, test_loader, device)
    
    # Calcular métricas finais
    from evaluate import evaluate
    auc_score, accuracy, f1, recall, precision = evaluate(model, test_loader, device)
    
    print(f"\n=== MÉTRICAS FINAIS ===")
    print(f"AUC: {auc_score:.4f}")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Gerar gráficos
    print("\nGerando gráficos...")
    
    # 1. Matriz de Confusão
    plot_confusion_matrix(true_labels, predictions)
    
    # 2. Curva ROC
    plot_roc_curve(true_labels, probs)
    
    # 3. Curva Precision-Recall
    plot_precision_recall_curve(true_labels, probs)
    
    # 4. Distribuição de Probabilidades
    plot_probability_distribution(true_labels, probs)
    
    print("\nTodos os gráficos foram gerados e salvos!")

if __name__ == "__main__":
    main()