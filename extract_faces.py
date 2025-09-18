from facenet_pytorch import MTCNN
import os, cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

# Configuração otimizada de dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

mtcnn = MTCNN(keep_all=True, min_face_size=20, thresholds=[0.5, 0.6, 0.7], device=device)

# Configuração de paralelismo
NUM_WORKERS = min(4, os.cpu_count() // 2)  # Ajusta automaticamente
print(f"Usando {NUM_WORKERS} workers para processamento paralelo")

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
                #print(f"No faces in frame {count} of {os.path.basename(video_path)}")
                count += 1
                continue

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

            #print(f"{len(face_batch)} faces in frame {count} of {os.path.basename(video_path)} detected")

            for face in face_batch:
                if face.dim() != 3:
                    continue

                face_np = (face.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')

                #print(f"Brightness: {np.mean(face_np):.2f}")
                #plt.imshow(face_np)
                #plt.axis("off")
                #plt.title("Extracted Face Preview")
                #plt.show()
                
                filename = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_f{saved}.jpg")
                #print(f"Save Loc: {filename}")
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
