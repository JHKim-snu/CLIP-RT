import json
import numpy as np
import os
from tqdm import tqdm

class ActionProcessor:
    def __init__(self, base_directory, min_threshold=0.01, angle_threshold=np.deg2rad(10)):
        self.base_directory = base_directory #base data directory 경로
        self.min_threshold = min_threshold #min position threshold
        self.angle_threshold = angle_threshold #min angle threshold

    def normalize_vector(self, action):
        positional_part = action[:3]  # x, y, z
        angular_part = action[3:6]  # roll, pitch, yaw
        gripper_value = action[6:] 
        
        # normalizing - positional part
        pos_norm = np.linalg.norm(positional_part)
        if pos_norm >= self.min_threshold:
            normalized_positional = (np.array(positional_part) / pos_norm).tolist()
        else:
            normalized_positional = positional_part
        
        # normalizing - angular part
        angular_norm = np.linalg.norm(angular_part)
        if angular_norm >= self.angle_threshold:
            normalized_angular = (np.array(angular_part) / angular_norm).tolist()
        else:
            normalized_angular = angular_part

        # 정규화된 벡터와 그리퍼 값을 결합하여 반환
        return normalized_positional + normalized_angular + gripper_value

    def process_files(self):
        for task_num in range(8):
            task_dir = os.path.join(self.base_directory, f"task_{task_num}")
            if not os.path.exists(task_dir):
                print(f"{task_dir} Passed...")
                continue
            
            # for all json files in the task dir 
            for root, _, files in os.walk(task_dir):
                for file_name in tqdm(files, desc=f"Procesisng {task_dir}"):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(root, file_name)
                        
                        with open(file_path, 'r') as file:
                            data = json.load(file)
                            
                            # normalizing
                            gt_action = data.get('action', [])
                            data['unitized_action'] = self.normalize_vector(gt_action)
                            
                            # update json 
                            with open(file_path, 'w') as output_file:
                                json.dump(data, output_file, indent=4)

# Example usage
base_directory_path = "./preprocess/data_0"
processor = ActionProcessor(base_directory_path)
processor.process_files()
