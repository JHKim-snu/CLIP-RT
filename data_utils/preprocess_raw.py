import numpy as np
import os
import json
from PIL import Image
import tensorflow_datasets as tfds

#### RIST ACTION CONVERSION
# this function converts 8-d action to 7-d action
# 8-d: [dx,dy,dz,dr,dp,dy,grp_rotation,grp_state]
# 7-d: [dx,dy,dz,dr,dp,dy,grp_state]

def rist_action_conversion(episode_path):
    data_files = sorted(os.listdir(episode_path))
    epi = []
    roll = 0
    pitch = 0
    yaw = 0
    
    for data_file in data_files:
        data_file_path = os.path.join(episode_path,data_file)

        if '.json' in data_file_path:
            with open(data_file_path,'r') as f:
                data = json.load(f)


            assert len(data['action']) == 8
            
            data['openx_action'] = data['action'][:6] + data['action'][-1:]
            
            if abs(data['action'][6]) > 0.3:
                if abs(roll) > 1.4: # over 80 degrees
                    data['openx_action'][4] = data['action'][6]
                elif abs(pitch) > 1.4: # over 80 degrees
                    data['openx_action'][3] = data['action'][6]
                elif abs(roll)<0.3 and abs(pitch)<0.3:
                    data['openx_action'][5] = data['action'][6]
                else:
                    print(f"CHECK {data_file_path} (nothing changed)")
                    
            if abs(data['action'][3]) > 0.3:
                roll += data['action'][3]
            if abs(data['action'][4]) > 0.3:
                pitch += data['action'][4]
            if abs(data['action'][5]) > 0.3:
                yaw += data['action'][5]
                

            with open(data_file_path, 'w') as fW:
                json.dump(data, fW)
            
                
if __name__ == "__main__":
    raw_root = ""  # raw data dir path 

    tasks = [f for f in os.listdir(raw_root) if not f.startswith('.')]
    for j, task in enumerate(tasks):
        task_root = os.path.join(raw_root,task)
        episodes = [f for f in os.listdir(task_root) if not f.startswith('.')]
        for i, episode in enumerate(episodes):
            episode_path = os.path.join(task_root,episode)
            try:
                rist_action_conversion(episode_path)
            except AssertionError as e:
                print("ASSERTION ERROR")
                print(episode_path)