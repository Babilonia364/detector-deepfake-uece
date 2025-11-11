import torch
from torchvision import transforms
from PIL import Image
from resnet_classifier import get_model
import os

def simple_predict(image_path, model_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carregar modelo
    model = get_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Pré-processar imagem
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predição
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        predicted_class = output.argmax().item()
    
    classes = ['real', 'fake']
    print(f"Imagem: {os.path.basename(image_path)}")
    print(f"Predição: {classes[predicted_class]}")
    print(f"Confiança: {prob[0][predicted_class]:.3f}")
    
    return classes[predicted_class], prob[0][predicted_class].item()

# Uso rápido
if __name__ == "__main__":
    simple_predict('./data/trained/real/899_f9.jpg')