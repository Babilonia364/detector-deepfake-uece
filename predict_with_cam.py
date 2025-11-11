#!/usr/bin/env python3
"""
Deepfake Detection with Grad-CAM Visualization
=============================================
Script para carregar modelo treinado e gerar mapas de calor para novas imagens
"""

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Importar módulos do projeto
from resnet_classifier import get_model
from grad_cam import analyze_predictions_with_cam, GradCAM

def load_trained_model(model_path='best_model.pth', device='cuda'):
    """Carrega o modelo treinado"""
    model = get_model()
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modelo carregado de {model_path}")
        print(f"   Melhor F1: {checkpoint.get('f1', 'N/A'):.4f}")
    else:
        print(f"❌ Arquivo do modelo não encontrado: {model_path}")
        return None
    
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, transform=None):
    """Pré-processa uma imagem para o modelo"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return input_tensor, original_image
    except Exception as e:
        print(f"❌ Erro ao carregar imagem {image_path}: {e}")
        return None, None

def predict_single_image(model, image_path, device, show_result=True):
    """Faz predição em uma única imagem e retorna resultado"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor, original_image = preprocess_image(image_path, transform)
    if input_tensor is None:
        return None
    
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    class_names = ['real', 'fake']  # Ajuste conforme suas classes
    result = {
        'predicted_class': predicted_class,
        'class_name': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities.cpu().numpy()[0]
    }
    
    if show_result:
        print(f"\n🎯 PREDIÇÃO PARA: {os.path.basename(image_path)}")
        print(f"   Classe: {result['class_name'].upper()}")
        print(f"   Confiança: {result['confidence']:.3f}")
        print(f"   Probabilidades: Real {result['probabilities'][0]:.3f}, Fake {result['probabilities'][1]:.3f}")
    
    return result, input_tensor, original_image

def generate_cam_for_image(model, image_path, output_dir="single_predictions", device='cuda'):
    """Gera Grad-CAM para uma imagem específica"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Fazer predição
    result, input_tensor, original_image = predict_single_image(model, image_path, device)
    if result is None:
        return
    
    # Configurar Grad-CAM
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    
    # Gerar mapas de calor
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_prefix = f"{base_name}_{result['class_name']}_conf_{result['confidence']:.3f}"
    
    try:
        # Grad-CAM padrão
        cam_standard, _ = gradcam.generate_cam(input_tensor, target_class=result['predicted_class'], method="gradcam")
        gradcam.visualize(input_tensor, cam_standard, 
                         save_path=os.path.join(output_dir, f"{save_prefix}_gradcam_cam.png"))
        
        # Grad-CAM++
        cam_plus, _ = gradcam.generate_cam(input_tensor, target_class=result['predicted_class'], method="gradcam++")
        gradcam.visualize(input_tensor, cam_plus,
                         save_path=os.path.join(output_dir, f"{save_prefix}_gradcam++_cam.png"))
        
        # Salvar imagem original
        if original_image:
            original_image.save(os.path.join(output_dir, f"{save_prefix}_original.png"))
        
        print(f"✅ Mapas de calor salvos em: {output_dir}/")
        print(f"   - {save_prefix}_original.png")
        print(f"   - {save_prefix}_gradcam_cam.png")
        print(f"   - {save_prefix}_gradcam++_cam.png")
        
    except Exception as e:
        print(f"❌ Erro ao gerar Grad-CAM: {e}")
    finally:
        gradcam.cleanup()
    
    return result

def batch_predict_with_cam(model, image_folder, output_dir="batch_predictions", device='cuda'):
    """Processa todas as imagens de uma pasta"""
    if not os.path.exists(image_folder):
        print(f"❌ Pasta não encontrada: {image_folder}")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    if not image_files:
        print(f"❌ Nenhuma imagem encontrada em {image_folder}")
        return
    
    print(f"\n🔍 Encontradas {len(image_files)} imagens em {image_folder}")
    
    results = []
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processando: {os.path.basename(image_path)}")
        result = generate_cam_for_image(model, image_path, output_dir, device)
        if result:
            results.append(result)
    
    # Resumo do batch
    if results:
        real_count = sum(1 for r in results if r['class_name'] == 'real')
        fake_count = sum(1 for r in results if r['class_name'] == 'fake')
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"\n📊 RESUMO DO BATCH:")
        print(f"   Total de imagens: {len(results)}")
        print(f"   Reais: {real_count}, Fakes: {fake_count}")
        print(f"   Confiança média: {avg_confidence:.3f}")
        print(f"   Resultados salvos em: {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection with Grad-CAM')
    parser.add_argument('--image', type=str, help='Caminho para uma única imagem')
    parser.add_argument('--folder', type=str, help='Caminho para pasta com imagens')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Caminho do modelo treinado')
    parser.add_argument('--output', type=str, default='predictions', help='Pasta de saída')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Usando dispositivo: {device}")
    
    # Carregar modelo
    model = load_trained_model(args.model, device)
    if model is None:
        return
    
    # Processar conforme os argumentos
    if args.image:
        print(f"\n🎯 Processando imagem única: {args.image}")
        generate_cam_for_image(model, args.image, args.output, device)
    
    elif args.folder:
        print(f"\n📁 Processando pasta: {args.folder}")
        batch_predict_with_cam(model, args.folder, args.output, device)
    
    else:
        print("❌ Especifique --image ou --folder")
        print("Exemplos de uso:")
        print("  python predict_with_cam.py --image caminho/para/imagem.jpg")
        print("  python predict_with_cam.py --folder caminho/para/pasta")
        print("  python predict_with_cam.py --image teste.jpg --output meus_resultados")

if __name__ == "__main__":
    main()