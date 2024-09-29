import json
import numpy as np
import os
from tqdm import tqdm

class AccumulatedActionProcessor:
    def __init__(self, base_directory):
        self.base_directory = base_directory

    def calculate_delta(self, current_action, next_action):
        """
        Calculate the delta between the current action and next action.
        """
        return np.array(next_action) - np.array(current_action)

    def process_files(self):
        """
        Process all JSON files in the task directories, calculating waypoint metrics.
        """
        # Iterate through task_0 to task_7 directories
        for task_num in range(8):
            task_dir = os.path.join(self.base_directory, f"task_{task_num}")
            if not os.path.exists(task_dir):
                print(f"{task_dir} does not exist. Skipping...")
                continue
            
            # Iterate through all JSON files in the task directory in sorted order (time order)
            for root, _, files in os.walk(task_dir):
                files = sorted(files)
                accumulated_action = np.zeros(8)  # Initialize for each task directory
                for i, file_name in enumerate(tqdm(files, desc=f"Processing {task_dir}")):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(root, file_name)
                        
                        with open(file_path, 'r') as file:
                            data = json.load(file)
                        
                        action = np.array(data.get('action', [0.0] * 8))  # Get the action (or fallback to zero array)

                        # Calculate accumulated action deltas
                        if i < len(files) - 1:
                            # Read the next file to calculate delta towards the waypoint
                            next_file_path = os.path.join(root, files[i + 1])
                            with open(next_file_path, 'r') as next_file:
                                next_data = json.load(next_file)
                                next_action = np.array(next_data.get('action', [0.0] * 8))
                                delta = self.calculate_delta(action, next_action)
                                accumulated_action += delta

                        # Add accumulated action to JSON
                        data['accumulated_action'] = accumulated_action.tolist()

                        # Calculate waypoint metric (remaining action to waypoint)
                        if data.get('waypoint', 0) == 1:
                            data['waypoint_metric'] = accumulated_action.tolist()
                            accumulated_action = np.zeros(8)  # Reset after reaching a waypoint

                        # Save updated JSON data back to the file
                        with open(file_path, 'w') as output_file:
                            json.dump(data, output_file, indent=4)

# Example usage
base_directory_path = ".//preprocess/data_3"
processor = AccumulatedActionProcessor(base_directory_path)
processor.process_files()
