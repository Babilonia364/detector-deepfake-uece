"""
Grad-CAM Implementation for Deepfake Detection
=============================================
Implementa Grad-CAM e Grad-CAM++ para visualizar regiões importantes
na classificação de deepfakes.

Funcionalidades:
- Grad-CAM padrão
- Grad-CAM++ (melhor localização)
- Geração de heatmaps sobrepostos na imagem
- Suporte para múltiplas imagens
"""

import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Registrar hooks - CORREÇÃO: usar hooks mais estáveis
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        # CORREÇÃO: usar register_backward_hook em vez de register_full_backward_hook
        self.backward_handle = self.target_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        # CORREÇÃO: substituir em vez de append para evitar acúmulo
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        # CORREÇÃO: capturar grad_output[0] que contém os gradientes das ativações
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image_tensor, target_class=None, method="gradcam"):
        """
        Gera mapa de calor CAM para uma imagem
        """
        self.model.eval()
        self.activations = None
        self.gradients = None
        
        # Forward pass
        output = self.model(image_tensor)
        predictions = torch.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        # CORREÇÃO: remover retain_graph=True para economizar memória
        class_score.backward()
        
        # Verificar se capturamos ativações e gradientes
        if self.activations is None:
            raise RuntimeError("Não foi possível capturar ativações.")
        if self.gradients is None:
            raise RuntimeError("Não foi possível capturar gradientes.")
        
        # Obter ativações e gradientes
        activation = self.activations
        gradient = self.gradients
        
        # Converter para numpy
        activation = activation.cpu().numpy()[0]  # Shape: (C, H, W)
        gradient = gradient.cpu().numpy()[0]      # Shape: (C, H, W)
        
        print(f"Debug - Activation shape: {activation.shape}, Gradient shape: {gradient.shape}")
        
        # Calcular pesos
        if method == "gradcam++":
            weights = self._compute_gradcam_plus_weights(gradient, activation)
        else:  # Grad-CAM padrão
            # Grad-CAM: média global dos gradientes para cada canal
            weights = np.mean(gradient, axis=(1, 2))  # Shape: (C,)
        
        print(f"Debug - Weights shape: {weights.shape}")
        
        # Gerar CAM
        cam = np.zeros(activation.shape[1:], dtype=np.float32)  # Shape: (H, W)
        for i, w in enumerate(weights):
            cam += w * activation[i, :, :]
        
        # Aplicar ReLU e normalizar
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, predictions.detach().cpu().numpy()[0]
    
    def _compute_gradcam_plus_weights(self, gradient, activation):
        """Calcula pesos para Grad-CAM++"""
        # Grad-CAM++: cálculo mais complexo dos pesos
        grad_2 = gradient ** 2
        grad_3 = gradient ** 3
        
        # Somar sobre altura e largura, mantendo os canais
        numerator = grad_2
        denominator = 2 * grad_2 + np.sum(activation * grad_3, axis=(1, 2), keepdims=True)
        
        # Evitar divisão por zero
        denominator = np.where(denominator != 0.0, denominator, 1e-10)
        alpha = numerator / denominator
        
        # Calcular pesos
        weights = np.sum(alpha * np.maximum(gradient, 0), axis=(1, 2))
        
        return weights
    
    def visualize(self, image_tensor, cam, save_path=None, alpha=0.4):
        """
        Visualiza CAM sobreposto na imagem original
        """
        # Pré-processar imagem original
        img = image_tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        
        # CORREÇÃO: Verificar se a normalização está correta para seu dataset
        # Se seu modelo foi treinado com normalização ImageNet padrão, use:
        img = np.clip((img * np.array([0.229, 0.224, 0.225]) + 
                      np.array([0.485, 0.456, 0.406])), 0, 1)
        
        # Se você usou outra normalização no treino, ajuste aqui:
        # Exemplo para normalização [0.5, 0.5, 0.5]:
        # img = np.clip((img * 0.5 + 0.5), 0, 1)
        
        # Redimensionar CAM para tamanho da imagem
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # Criar heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Sobrepor heatmap na imagem
        superimposed_img = heatmap * alpha + img
        superimposed_img = np.clip(superimposed_img, 0, 1)
        
        if save_path:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            
            # Salvar imagem sobreposta
            cv2.imwrite(save_path, np.uint8(255 * superimposed_img[:, :, ::-1]))
            
            # Salvar imagem original também
            orig_path = save_path.replace('_cam.png', '_orig.png')
            cv2.imwrite(orig_path, np.uint8(255 * img[:, :, ::-1]))
        
        return superimposed_img
    
    def cleanup(self):
        """Remove hooks"""
        self.forward_handle.remove()
        self.backward_handle.remove()


def analyze_predictions_with_cam(model, test_data, device, num_examples=5, output_dir="ignore/gradcam_results", target_layer_name=None):
    """
    Função principal para analisar predições com Grad-CAM
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # CORREÇÃO: Tentar diferentes camadas alvo, começando por uma menos profunda
    if target_layer_name:
        # Se o usuário especificou uma camada, tente usá-la
        try:
            target_layer = eval(f"model.{target_layer_name}")
            print(f"Usando camada alvo especificada: {target_layer_name}")
        except:
            print(f"Camada {target_layer_name} não encontrada. Usando fallback.")
            target_layer_name = None
    
    if not target_layer_name:
        # Tentar camadas progressivamente menos profundas
        layer_candidates = [
            "layer3[-1]",  # CORREÇÃO: Começar com layer3 em vez de layer4
            "layer2[-1]",  # Opção mais rasa para melhor localização
            "layer4[-1]",  # Original (mais profunda)
            "features[-1]",  # Para outros tipos de modelo
        ]
        
        for layer_candidate in layer_candidates:
            try:
                target_layer = eval(f"model.{layer_candidate}")
                print(f"Usando camada alvo: model.{layer_candidate}")
                break
            except AttributeError:
                continue
        else:
            print("Erro: Não foi possível encontrar uma camada alvo adequada.")
            return []
    
    gradcam = GradCAM(model, target_layer)
    
    print(f"\nAnalisando {num_examples} exemplos com Grad-CAM...")
    
    results = []
    successful_examples = 0
    
    for idx in range(min(num_examples, len(test_data))):
        try:
            # Carregar imagem
            img, true_label = test_data[idx]
            input_tensor = img.unsqueeze(0).to(device)
            
            print(f"\n--- Processando Exemplo {idx+1} ---")
            
            # Gerar Grad-CAM
            cam_standard, probs = gradcam.generate_cam(input_tensor, method="gradcam")
            print(f"Grad-CAM gerado com shape: {cam_standard.shape}")
            
            # Gerar Grad-CAM++
            cam_plus, _ = gradcam.generate_cam(input_tensor, method="gradcam++")
            print(f"Grad-CAM++ gerado com shape: {cam_plus.shape}")
            
            # Obter predição
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            
            # Nomes das classes
            true_label_name = test_data.classes[true_label]
            pred_label_name = test_data.classes[pred_class]
            
            # Salvar visualizações
            base_name = f"exemplo_{idx+1}_real_{true_label_name}_pred_{pred_label_name}_conf_{confidence:.3f}"
            
            # Grad-CAM padrão
            gradcam.visualize(input_tensor, cam_standard, 
                             save_path=os.path.join(output_dir, f"{base_name}_gradcam_cam.png"))
            
            # Grad-CAM++
            gradcam.visualize(input_tensor, cam_plus,
                             save_path=os.path.join(output_dir, f"{base_name}_gradcam++_cam.png"))
            
            # Salvar imagem original
            orig_img = img.permute(1, 2, 0).detach().numpy()
            orig_img = np.clip((orig_img * np.array([0.229, 0.224, 0.225]) + 
                               np.array([0.485, 0.456, 0.406])), 0, 1)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_orig.png"), 
                       np.uint8(255 * orig_img[:, :, ::-1]))
            
            # Coletar resultados
            results.append({
                'example_id': idx + 1,
                'true_label': true_label_name,
                'pred_label': pred_label_name,
                'confidence': confidence,
                'correct': true_label == pred_class
            })
            
            print(f"✓ Exemplo {idx+1} processado: Real={true_label_name}, Predito={pred_label_name} "
                  f"(conf: {confidence:.3f}) {'✓' if true_label == pred_class else '✗'}")
            
            successful_examples += 1
            
        except Exception as e:
            print(f"✗ Erro no exemplo {idx+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Limpar hooks
    gradcam.cleanup()
    
    # Gerar relatório resumido
    if successful_examples > 0:
        correct_predictions = sum(1 for r in results if r['correct'])
        accuracy = correct_predictions / successful_examples
        
        print(f"\n Resumo da análise:")
        print(f"Exemplos processados com sucesso: {successful_examples}/{num_examples}")
        print(f"Precisão nos exemplos: {accuracy:.1%} ({correct_predictions}/{successful_examples})")
        print(f"Mapas salvos em: {output_dir}/")
        
        # Mostrar preview dos arquivos gerados
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"Arquivos gerados: {len(files)}")
            for file in files[:5]:  # Mostrar primeiros 5 arquivos
                print(f"  - {file}")
    else:
        print("\n Nenhum exemplo foi processado com sucesso.")
        print("Possíveis causas:")
        print("1. Problema com a camada alvo do modelo")
        print("2. Dimensões incompatíveis das ativações")
        print("3. Erro no hook de backward")
    
    return results