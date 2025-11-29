"""
Predict with Grad-CAM - Deepfake Detection
==========================================
Script simplificado para predições com Grad-CAM em imagens individuais
Reutiliza exatamente o mesmo código da main.py
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import sys

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grad_cam import analyze_predictions_with_cam
from resnet_classifier import get_model

class SingleImageDataset:
    """Dataset fake para uma única imagem - compatível com analyze_predictions_with_cam"""
    def __init__(self, image_tensor, transform):
        self.image_tensor = image_tensor
        self.transform = transform
        self.classes = ['real', 'fake']  # mesma ordem do treinamento
    
    def __getitem__(self, index):
        # Retorna a imagem transformada e um label dummy (0 = real)
        return self.image_tensor, 0
    
    def __len__(self):
        return 1

def predict_single_image(image_path, model_path='best_model.pth', output_dir='gradcam_predictions'):
    """
    Faz predição com Grad-CAM para uma única imagem
    Reutiliza exatamente a mesma lógica da main.py
    """
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Carregar modelo (igual na main.py)
    model = get_model()
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"✓ Modelo carregado: {model_path}")
    else:
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    # Pré-processamento (igual ao validation da main.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    # Carregar e pré-processar imagem
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension
    
    print(f"✓ Imagem processada: {image_path}")
    
    # Criar dataset fake compatível com analyze_predictions_with_cam
    test_data = SingleImageDataset(image_tensor.squeeze(0), transform)
    
    # Usar EXATAMENTE a mesma função da main.py
    print("\n" + "="*60)
    print("GERANDO GRAD-CAM")
    print("="*60)
    
    results = analyze_predictions_with_cam(
        model=model,
        test_data=test_data,
        device=device,
        num_examples=1,
        output_dir=output_dir,
        target_layer_name="layer3[-1]"
    )
    
    print("\n✓ Análise concluída!")
    print(f"✓ Resultados salvos em: {output_dir}/")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Predição com Grad-CAM para Deepfake Detection')
    parser.add_argument('--image', type=str, required=True, help='Caminho para a imagem de entrada')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Caminho para o modelo treinado')
    parser.add_argument('--output', type=str, default='ignore/gradcam_predictions', help='Diretório de saída')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"✗ Erro: Imagem não encontrada: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"✗ Erro: Modelo não encontrado: {args.model}")
        print("Certifique-se de que best_model.pth está na raiz do projeto")
        return
    
    try:
        predict_single_image(
            image_path=args.image,
            model_path=args.model,
            output_dir=args.output
        )
    except Exception as e:
        print(f"✗ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()