import json
import numpy as np
import os
from tqdm import tqdm
import json
import numpy as np
import os
from tqdm import tqdm

class AccumulatedActionProcessor:
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.accumulated_action = np.zeros(8)  # Initialize accumulated action with zeros for [x, y, z, roll, pitch, yaw, gripper, last element]

    def accumulate_action(self, action):
        self.accumulated_action += np.array(action)

    def reset_accumulated_action(self):
        self.accumulated_action = np.zeros(8)

    def process_files(self):
        # Iterate through task_0 to task_7 directories
        for task_num in range(8):
            task_dir = os.path.join(self.base_directory, f"task_{task_num}")
            if not os.path.exists(task_dir):
                print(f"{task_dir} does not exist. Skipping...")
                continue
            
            # Iterate through all JSON files in the task directory in sorted order (time order)
            for root, _, files in os.walk(task_dir):
                files = sorted(files)  # Sort files to ensure time order
                first_file = True  # Flag to set accumulated_action to zero for the first file
                for file_name in tqdm(files, desc=f"Processing {task_dir}"):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(root, file_name)
                        
                        with open(file_path, 'r') as file:
                            data = json.load(file)
                            
                            # If it's the first file, initialize accumulated_action to zeros, then accumulate the first action
                            if first_file:
                                self.reset_accumulated_action()
                                action = data.get('action', [])
                                if len(action) == 8:  # Check to ensure the action array has the correct length
                                    self.accumulate_action(action)
                                data['accumulated_action'] = self.accumulated_action.tolist()
                                first_file = False
                            else:
                                # Accumulate action values
                                action = data.get('action', [])
                                if len(action) == 8:  # Check to ensure the action array has the correct length
                                    self.accumulate_action(action)
                                data['accumulated_action'] = self.accumulated_action.tolist()
                            
                            # Check if waypoint is 1, reset accumulated action but still accumulate current action
                            if data.get('waypoint', 0) == 1:
                                print(f"Waypoint! Resetting accumulated action for {file_name}.")
                                self.reset_accumulated_action()
                                if len(action) == 8:  # Accumulate the current action after resetting
                                    self.accumulate_action(action)
                                data['accumulated_action'] = self.accumulated_action.tolist()

                            # Save updated JSON data back to the file
                            with open(file_path, 'w') as output_file:
                                json.dump(data, output_file, indent=4)

# Example usage
base_directory_path = "./preprocess/data_0"
processor = AccumulatedActionProcessor(base_directory_path)
processor.process_files()
