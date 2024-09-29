import json
import os
from tqdm import tqdm

class ActionProcessor:
    def __init__(self, base_directory):
        """
        작업 폴더를 포함하는 기본 디렉토리로 ActionProcessor를 초기화합니다.

        Args:
            base_directory (str): 작업 폴더(task_0에서 task_7까지)를 포함하는 기본 디렉토리 경로.
        """
        self.base_directory = base_directory

    def remove_unified_action(self, data):
        """
        JSON 데이터에서 unified_action 필드를 삭제합니다.

        Args:
            data (dict): JSON 데이터.

        Returns:
            dict: unified_action 필드가 삭제된 JSON 데이터.
        """
        if 'unified_action' in data:
            del data['unified_action']
        return data

    def process_files(self):
        """
        작업 디렉토리의 모든 JSON 파일을 처리하여 unified_action 필드를 제거합니다.
        """
        # task_0에서 task_7 디렉토리를 반복합니다.
        for task_num in range(8):
            task_dir = os.path.join(self.base_directory, f"task_{task_num}")
            if not os.path.exists(task_dir):
                print(f"디렉토리 {task_dir}가 존재하지 않습니다. 건너뜁니다...")
                continue
            
            # 작업 디렉토리의 모든 JSON 파일을 반복합니다.
            for root, _, files in os.walk(task_dir):
                for file_name in tqdm(files, desc=f"{task_dir} 처리 중"):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(root, file_name)
                        
                        with open(file_path, 'r') as file:
                            data = json.load(file)
                            
                            # JSON 데이터에서 unified_action 필드를 삭제합니다.
                            data = self.remove_unified_action(data)
                            
                            # 업데이트된 JSON 데이터를 파일에 다시 저장합니다.
                            with open(file_path, 'w') as output_file:
                                json.dump(data, output_file, indent=4)

# Example usage
base_directory_path = ".//preprocess/data_0 copy"
processor = ActionProcessor(base_directory_path)
processor.process_files()
