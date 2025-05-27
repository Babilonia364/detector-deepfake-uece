import cv2
import os

def extract_frames(video_path, output_dir, num_frames=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames

    count, saved = 0, 0
    while cap.isOpened() and saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_f{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    cap.release()
