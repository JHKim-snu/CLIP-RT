import json
import numpy as np
import os
from tqdm import tqdm

class RemainedActionProcessor:
    def __init__(self, base_directory):
        self.base_directory = base_directory

    def process_files(self):
        """
        Process all JSON files in the given directory to calculate remained_action.
        """
        for task_num in range(8):
            task_dir = os.path.join(self.base_directory, f"task_{task_num}")
            if not os.path.exists(task_dir):
                print(f"{task_dir} does not exist. Skipping...")
                continue

            for root, _, files in os.walk(task_dir):
                files = sorted(files)

                # Iterate through each file and calculate remained_action
                for file_name in tqdm(files, desc=f"Processing {task_dir}"):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(root, file_name)

                        try:
                            with open(file_path, 'r', encoding='utf-8') as file:
                                data = json.load(file)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            print(f"Error reading {file_path}, skipping...")
                            continue
                        
                        # Extract waypoint_action and accumulated_action
                        waypoint_action = np.array(data.get('waypoint_accum', [0.0] * 8))
                        accumulated_action = np.array(data.get('accumulated_action', [0.0] * 8))

                        # Calculate remained_action
                        remained_action = waypoint_action - accumulated_action
                        data['remained_action'] = remained_action.tolist()

                        # Save updated JSON data back to the file
                        with open(file_path, 'w') as output_file:
                            json.dump(data, output_file, indent=4)
                            # print(f"remained_action added to {file_name}")

# Example usage
base_directory_path = "./preprocess/data_0"
processor = RemainedActionProcessor(base_directory_path)
processor.process_files()