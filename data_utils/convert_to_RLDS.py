import numpy as np
import os
import json
from PIL import Image
import tensorflow_datasets as tfds

raw_root = "" # raw data dir path

image_size = 256  # we crop the left 256 pixels of the image

os.makedirs('./train', exist_ok=True)
os.makedirs('./test', exist_ok=True)

tasks = [f for f in os.listdir(raw_root) if not f.startswith('.')]

for j, task in enumerate(tasks):
    task_root = os.path.join(raw_root,task)
    episodes = [f for f in os.listdir(task_root) if not f.startswith('.')]
    for i, episode in enumerate(episodes):
        episode_path = os.path.join(task_root,episode)
        data_files = os.listdir(episode_path)
        epi = []
        for data_file in data_files:
            data_file_path = os.path.join(episode_path,data_file)
            
            if '.json' in data_file_path:
                with open(data_file_path,'r') as f:
                    data = json.load(f)
                image_path = os.path.join(episode_path, data_file_path.split(".json")[0].split("/")[-1]+".png")
                with Image.open(image_path) as img:
                    img = img.resize((image_size, image_size))
                    img_array = np.array(img, dtype=np.uint8)
                    if img_array.shape != (image_size, image_size, 3):
                        print(image_path)
                        raise ValueError(f"Unexpected array shape: {img_array.shape}. Expected shape is ({image_size}, {image_size}, 3).")
                action = data['openx_action']
                assert len(action) == 7
                epi.append({
                    'image': img_array,
                    'action': np.asarray(np.array(action), dtype=np.float32),
                    'language_instruction': data['instruction']
                })
        if i != 0:
            np.save(f'test/task_{j}_episode_{i}.npy', epi)
            np.save(f'train/task_{j}_episode_{i}.npy', epi)
