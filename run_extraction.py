from extract_frames import extract_frames
import os
from tqdm import tqdm

def process_all(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for video_file in tqdm(os.listdir(input_folder)):
        in_path = os.path.join(input_folder, video_file)
        out_path = os.path.join(output_folder, video_file.split('.')[0])
        extract_frames(in_path, out_path)

process_all("data/train/real", "frames/train/real")
process_all("data/train/fake", "frames/train/fake")
process_all("data/test/real", "frames/test/real")
process_all("data/test/fake", "frames/test/fake")