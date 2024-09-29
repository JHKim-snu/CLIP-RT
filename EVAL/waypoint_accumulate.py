import json
import numpy as np
import os
from tqdm import tqdm

class FinalAccumProcessor:
    def __init__(self, base_directory):
        self.base_directory = base_directory

    def process_files(self):
        """
        Process all JSON files to find each waypoint = 1 and update previous files with waypoint_accum.
        """
        for task_num in range(8):
            task_dir = os.path.join(self.base_directory, f"task_{task_num}")
            if not os.path.exists(task_dir):
                print(f"{task_dir} does not exist. Skipping...")
                continue

            files = sorted(os.listdir(task_dir))
            accumulated_action = np.zeros(8)  # Initialize accumulated action
            waypoint_segments = []  # List to store segments of files between waypoints

            current_segment = []  # To track files before the next waypoint = 1
            for i, file_name in enumerate(tqdm(files, desc=f"Finding waypoints in {task_dir}")):
                if file_name.endswith('.json'):
                    file_path = os.path.join(task_dir, file_name)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        print(f"Error reading {file_path}, skipping...")
                        continue
                    
                    # Accumulate actions
                    action = np.array(data.get('action', [0.0] * 8))
                    accumulated_action += action
                    data['accumulated_action'] = accumulated_action.tolist()

                    # Track files and reset if waypoint = 1
                    current_segment.append((file_path, accumulated_action.copy()))
                    
                    if data.get('waypoint', 0) == 1:
                        waypoint_segments.append(current_segment)
                        current_segment = []  # Reset for the next segment
                        accumulated_action = np.zeros(8)  # Reset accumulated action

                    # Save the updated file
                    with open(file_path, 'w') as output_file:
                        json.dump(data, output_file, indent=4)

            # Process each segment to add waypoint_accum based on last accumulated value before each waypoint = 1
            for segment in waypoint_segments:
                self.update_files_with_waypoint_accum(segment)

    def update_files_with_waypoint_accum(self, segment):
        """
        Update given segment files with the waypoint_accum based on the last accumulated action before waypoint = 1.
        """
        # Use the last accumulated action of the segment as waypoint_accum for all files in the segment
        waypoint_accum = segment[-1][1] if segment else np.zeros(8)
        
        for file_path, _ in segment:  # Apply to all files including the waypoint = 1
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            except (UnicodeDecodeError, json.JSONDecodeError):
                print(f"Error reading {file_path}, skipping...")
                continue

            data['waypoint_accum'] = waypoint_accum.tolist()  # Add waypoint_accum to each file

            with open(file_path, 'w') as output_file:
                json.dump(data, output_file, indent=4)
                # print(f"waypoint_accum updated for {file_path}")

# Example usage
base_directory_path = "./preprocess/data_0"
processor = FinalAccumProcessor(base_directory_path)
processor.process_files()
