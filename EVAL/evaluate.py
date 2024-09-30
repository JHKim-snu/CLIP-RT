import numpy as np
import json
import os
class RobotEvaluator:
    def __init__(self, output_directory):
        self.output_directory = output_directory

    def evaluate(self, predicted_action, json_file_path):
        # Ensure predicted_action is a numpy array
        predicted_action = np.array(predicted_action)

        # Read data from JSON file
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except (UnicodeDecodeError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {json_file_path}: {e}. Skipping evaluation.")
            return

        # Calculate metrics
        l1_result = self.l1_distance(data, predicted_action)
        tf_result = self.tf_metric(data, predicted_action)
        accuracy_result = self.accuracy(data, predicted_action)

        # Save metrics
        self.save_metrics(json_file_path, l1_result, tf_result, accuracy_result)
        
        return l1_result, tf_result, accuracy_result
    
    def unitize_action(self, action): # unitize
        # Define threshold
        thres_xyz = 0.001  # xyz 0.1cm
        thres_rpyg = np.deg2rad(10)  # rpy 10 degree
        unitized_action = np.zeros_like(action)
        
        # Unitize 8dim 
        if len(action) == 8:
            unitized_action[:3] = np.where(np.abs(action[:3]) < thres_xyz, 0, np.sign(action[:3]))
            unitized_action[3:7] = np.where(np.abs(action[3:7]) < thres_rpyg, 0, np.sign(action[3:7]))
        # unitize 7dim
        else:
            unitized_action[:3] = np.where(np.abs(action[:3]) < thres_xyz, 0, np.sign(action[:3]))
            unitized_action[3:6] = np.where(np.abs(action[3:6]) < thres_rpyg, 0, np.sign(action[3:6]))
        
        return unitized_action

    def l1_distance(self, data, predicted_action):
        """
        Calculate the L1 distance between the unitized action and the predicted unitized action.
        """
        if len(predicted_action) == 8:
            unitized_action = self.unitize_action(np.array(data.get('action', [0.0] * 8)))
        else:
            unitized_action = self.unitize_action(np.array(data.get('openx_action', [0.0] * 7)))
        predicted_unitized_action = self.unitize_action(predicted_action)  # Unitize predicted action
        
        l1_dist = np.linalg.norm(unitized_action - predicted_unitized_action, ord=1)

        # Determine if the actions match
        result = True if l1_dist == 0 else False
        # print(f"L1 Distance: {l1_dist}, Result: {result}")

        return result # {"l1_distance": l1_dist, "match": result}

    def tf_metric(self, data, predicted_action):
        """
        Calculate the TF metric for the predicted action.
        If the predicted action is between the zero vector and the upper bound, return True.
        """
        # Get the current action from the data
        if len(predicted_action) == 8:
            now_action = np.array(data.get('action', [0.0] * 8))
            remained_action = np.array(data.get('remained_action', [0.0] * 8))
            zero_action = np.zeros(8)
        else:
            now_action = np.array(data.get('openx_action', [0.0] * 7))
            remained_action = np.array(data.get('openx_remained_action', [0.0] * 7))
            zero_action = np.zeros(7)
        # now_action = np.array(data.get('action', [0.0] * 8))
        # remained_action = np.array(data.get('remained_action', [0.0] * 8))
        
        # 0000000이면 true인데 만일 바로 전 action과 현재 action이랑 변함없으면 오답 : xyzrpyg = [0]이고 [-1]도 같으면  

        if np.linalg.norm(now_action[:7]) != 0: # x y z r p y g
            upper_bound = remained_action + now_action
            lower_bound = zero_action
            
            # Check if the predicted action is within the bounds
            is_correct1 = np.all((predicted_action >= lower_bound) & (predicted_action <= upper_bound))
            result1 = True if is_correct1 else False

            # Unitize the upper bound and current action
            unitized_upper_bound = self.unitize_action(upper_bound)
            unitized_now_action = self.unitize_action(now_action)
            
            # Check if the unitized current action is within the unitized bounds
            is_correct2 = np.all((unitized_now_action >= lower_bound) & (unitized_now_action <= unitized_upper_bound))
            result2 = True if is_correct2 else False

            # Combine the results
            result = result1 or result2

            # Print the results for debugging
            # print(f"result1: {result1}, result2: {result2}")
            # print(f"TF Metric: {result}")
            
            return result # {"tf_metric": result}
        
        else:  # now_action이 모두 0일 때
            # 이전 액션이 현재 액션과 동일한지 확인

            is_same_action = True if np.array_equal(now_action[-1], predicted_action[-1]) else False
            
            # 동일한 액션이라면 False 처리 (오답으로 간주)
            if is_same_action:
                return False # {"tf_metric": False}
            else:
                return True # {"tf_metric": True}



    def accuracy(self, data, predicted_action):
        
        # check its dimension
        if len(predicted_action) == 8:
            now_action = np.array(data.get('action', [0.0] * len(predicted_action)))
        else :
            now_action = np.array(data.get('openx_action', [0.0] * len(predicted_action)))
        
        if np.linalg.norm(now_action[:7]) != 0: 
            # Unitize actions
            unitized_now_action = self.unitize_action(now_action)
            unitized_predicted_action = self.unitize_action(predicted_action)
            xyz_tf = True if np.array_equal(unitized_now_action[:3], unitized_predicted_action[:3]) else False
            
            if len(predicted_action) == 8:
                rpyg_tf = True if np.array_equal(unitized_now_action[3:7], unitized_predicted_action[3:7]) else False
            else:
                rpyg_tf = True if np.array_equal(unitized_now_action[3:6], unitized_predicted_action[3:6]) else False

        else: # np.linalg.norm(now_action[:7]) == 0, zero input일 때 
            xyz_tf = False
            rpyg_tf = False

        # Gripper Status
        predicted_action[-1] = 1.0 if predicted_action[-1] > 0.5 else 0.0
        gripper_io = (now_action[-1] == predicted_action[-1])

        # Combine all accuracy results
        # print(f"XYZ Accuracy: {xyz_tf:.2f}%, \nRPYG Accuracy: {rpyg_tf:.2f}%, \nGripper Status Accuracy: {gripper_io}%")
        
        return {
            "XYZ Accuracy": xyz_tf, 
            "RPYG Accuracy": rpyg_tf, 
            "Gripper Status Accuracy": gripper_io,
        }

    def save_metrics(self, json_file_path, l1_result, tf_result, accuracy_result):
        # Convert numpy booleans to standard Python booleans
        def convert_np_to_python(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # For converting numpy arrays to lists
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        metrics = {
            "l1_distance": l1_result,
            "tf_metric": tf_result,
            "accuracy": accuracy_result
        }

        # If output_directory is None, print the metrics instead of saving
        if self.output_directory is None:
            try:
                # Use custom conversion function to handle non-serializable types
                print(f"Metrics for {json_file_path}:\n", json.dumps(metrics, indent=4, default=convert_np_to_python))
            except TypeError as e:
                print(f"Error serializing metrics: {e}")
        else:
            # Define the metrics file path with 'eval_' prefix
            metrics_file_name = f"eval_{os.path.basename(json_file_path)}"
            metrics_file_path = os.path.join(self.output_directory, metrics_file_name)

            try:
                with open(metrics_file_path, 'w') as metrics_file:
                    json.dump(metrics, metrics_file, indent=4, default=convert_np_to_python)
                    print(f"Metrics saved for {json_file_path}")
            except IOError as e:
                print(f"Error saving metrics for {json_file_path}: {e}")


if __name__ == "__main__":
    
    # Example usage with saving to the desired directory
    # output_directory_path = ".//to_eval"  # Output directory for saved metrics
    # evaluator = RobotEvaluator(output_directory_path)
    evaluator = RobotEvaluator(None)

    predicted_action = [0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Predicted action vector
    json_file_path = "./data_evaluation/known/data_1/task_0/2024_8_1_14_15_50.json"  # JSON file path

    # Perform the metric evaluation and save the results
    evaluator.evaluate(predicted_action, json_file_path)
