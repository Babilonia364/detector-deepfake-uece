from facenet_pytorch import MTCNN
import os, cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

mtcnn = MTCNN(keep_all=True, min_face_size=20, thresholds=[0.5, 0.6, 0.7], device='cuda')

def extract_faces_from_video(video_path, output_dir, num_faces=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // num_faces)

    count, saved = 0, 0
    while cap.isOpened() and saved < num_faces:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            faces = mtcnn(img_pil)

            if faces is None or (isinstance(faces, list) and len(faces) == 0):
                #print(f"No faces in frame {count} of {os.path.basename(video_path)} ")
                count += 1
                continue

            # Normalize to list of face tensors
            if isinstance(faces, torch.Tensor) and len(faces.shape) == 3:
                face_batch = [faces]
            elif isinstance(faces, torch.Tensor) and len(faces.shape) == 4:
                face_batch = [f for f in faces]
            elif isinstance(faces, list):
                face_batch = faces
            else:
                print(f"Unrecognized face format in frame {count}")
                count += 1
                continue

            #print(f" Detected {len(face_batch)} face(s) in frame {count} of {os.path.basename(video_path)}")

            for face in face_batch:
                if face.dim() != 3:
                    continue

                face_np = (face.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')

                #print(f"Brightness: {np.mean(face_np):.2f}")

                # Optional preview (just show one)
                #plt.imshow(face_np)
                #plt.axis("off")
                #plt.title("Extracted Face Preview")
                #plt.show()
                
                filename = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_f{saved}.jpg")
                #print(f"Saving to: {filename}")
                cv2.imwrite(filename, cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))
                saved += 1

        count += 1
    cap.release()

def extract_all_faces(input_root, output_root):
    for label in ['real', 'fake']:
        in_dir = os.path.join(input_root, label)
        out_dir = os.path.join(output_root, label)
        os.makedirs(out_dir, exist_ok=True)
        for video in tqdm(os.listdir(in_dir), desc=f"Processing {label}"):
            extract_faces_from_video(
                os.path.join(in_dir, video),
                out_dir
            )
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

device = torch.device('cuda') #or cpu or your favoured device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

extract_all_faces("video loc", "target loc") #Add your own adress

import os
data_dir = "adress where faces are stored"
train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
test_data = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
    print(f"Accuracy: {acc:.4f} | AUC: {auc_score:.4f}")
    plot_auc_roc(all_labels, all_probs)
    return auc_score, acc

for epoch in range(12):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to('cpu'), labels.to('cpu')
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    auc, acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1} Loss: {running_loss:.2f} AUC: {auc:.4f} Acc: {acc:.4f}")
