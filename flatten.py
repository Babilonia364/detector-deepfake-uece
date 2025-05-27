import os
import shutil

def flatten_frames(input_root):
    for label in ['real', 'fake']:
        label_dir = os.path.join(input_root, label)
        for folder in os.listdir(label_dir):
            subfolder = os.path.join(label_dir, folder)
            if os.path.isdir(subfolder):
                for file in os.listdir(subfolder):
                    src = os.path.join(subfolder, file)
                    dst = os.path.join(label_dir, f"{folder}_{file}")
                    shutil.move(src, dst)
                os.rmdir(subfolder)

flatten_frames('frames/train')
flatten_frames('frames/test')