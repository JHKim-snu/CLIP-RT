import os
import json
import numpy as np
import argparse
import random
import time

class ActionAugmentation:
    def __init__(self):
        self.dir = ""

    def calculate_action_sum(self, episode_path):
        actions = []
        action_sum = [0,0,0,0,0,0,0,0]
        
        # JSON read all
        json_files = sorted([f for f in os.listdir(episode_path) if f.endswith('.json')])

        # print(json_files)
        # return
    
        for json_file in json_files:
            with open(os.path.join(episode_path, json_file), 'r') as f:
                data = json.load(f)
                if 'action' in data:
                    action_sum = [a + b for a, b in zip(action_sum, data['action'])]
                    
        return action_sum
    
    def calculate_generated_action_sum(self, trajectory):
        action_sum = [0,0,0,0,0,0,0,0]
        for action in trajectory:
            action_sum = [a + b for a, b in zip(action_sum, action[0])]
        return action_sum

            
    def calculate_action_values(self, episode_path):
        actions = []
        parsed_traj = []
        
        # JSON read all
        json_files = sorted([f for f in os.listdir(episode_path) if f.endswith('.json')])

        # print(json_files)
        # return
    
        for json_file in json_files:
            with open(os.path.join(episode_path, json_file), 'r') as f:
                data = json.load(f)
                if 'action' in data:
                    actions.append(data['action'])

        # print(actions)
        # print("\n")
        
        base_step = 0
        num_actions_inchunk = 0
        for i in range(1, len(actions)):
            num_actions_inchunk += 1
            if ((actions[i][-1] != actions[i - 1][-1]) and i != base_step) or i == len(actions)-1: # change in gripper state or end of episode
                temp_chunk = {}
                if base_step < i-2:
                    action_sum = np.sum(actions[base_step:i - 2], axis=0).tolist() if i > 1 else []                    
                    temp_chunk["action_sum"] = action_sum
                    temp_chunk["fine_action_0"] = actions[i-2] 
                    temp_chunk["fine_action_1"] = actions[i-1] 
                    temp_chunk["fine_action_2"] = actions[i] 
                    temp_chunk["gripper"] = temp_chunk["fine_action_0"][-1]
                else: # if too short chunk of actinos, e.g., gripper change state in i=1
                    temp_chunk["actions"] = []
                    for k in range(base_step, i+1):
                        temp_chunk["action_sum"] = None
                        temp_chunk["actions"].append(actions[k])
                    temp_chunk["gripper"] = temp_chunk["actions"][0][-1]
                        
                parsed_traj.append(temp_chunk)
                base_step = i+1

        return parsed_traj

    def trajectory_sampler(self, delta_coord, seed=0):
        random.seed(seed)
        delta_xyz = delta_coord[:3]
        delta_rpyrot = delta_coord[3:7]
        delta_grp = 0 if delta_coord[-1]==0 else 1
        
        sampled_vectors = []
        sample_sizes = [0, 0.01, 0.05, 0.1]
        cnt = 0

        while not np.allclose(delta_xyz, [0, 0, 0]):
            cnt += 1
            sampled = []

            for i in range(3):
                if delta_xyz[i] < 0:
                    valid_samples = [-size for size in sample_sizes if size <= abs(delta_xyz[i])]
                    sampled_value = random.choice(valid_samples) if valid_samples else 0
                else:
                    valid_samples = [size for size in sample_sizes if size <= delta_xyz[i]]
                    sampled_value = random.choice(valid_samples) if valid_samples else 0
                
                sampled.append(sampled_value)

            delta_xyz = [delta_xyz[j] - sampled[j] for j in range(3)] 
            if sampled != [0,0,0]:
                sampled_vectors.append(sampled)
            if cnt > 100:
                break

        return sampled_vectors
    
    def try_all_generator(self, fine_action):
        try_all = []
        choices = [0.1, 0.05, 0.01, -0.01, -0.05, -0.1]
        if np.allclose(fine_action[0:3], [0,0,0]):
            try_all = [(fine_action, True)]
        else:
            for i, xyz in enumerate(fine_action[0:3]):
                false_action = [0,0,0]
                true_action = [0,0,0]
                random_act = random.choice(choices)
                if xyz == 0:
                    false_action[i] = random_act
                    true_action[i] = -random_act
                else:
                    false_action[i] = -fine_action[i]
                    true_action[i] = fine_action[i]
                try_all.append((false_action + [0,0,0,0] + [fine_action[-1]], False))
                try_all.append((true_action + [0,0,0,0] + [fine_action[-1]], True))
                    
        return try_all
    
    def trajectory_generator(self, episode_path):
        trajectory = []
        parsed_traj = self.calculate_action_values(episode_path)

        for temp_chunk in parsed_traj:
            if temp_chunk.get("action_sum") is not None:
                seed = int(time.time() * 1000) % 1000000
                sampled_xyz = self.trajectory_sampler(delta_coord=temp_chunk["action_sum"], seed=seed)
                # print(sampled_xyz)
                trajectory = trajectory + [(xyz+[0,0,0,0,temp_chunk["gripper"]], True) for xyz in sampled_xyz]
                # print(trajectory)
                if not np.allclose(temp_chunk["action_sum"][3:7], [0,0,0,0]):
                    trajectory.append(([0,0,0] + temp_chunk["action_sum"][3:7] + [temp_chunk["gripper"]], True))
                trajectory = trajectory + self.try_all_generator(temp_chunk["fine_action_0"])
                trajectory = trajectory + [(temp_chunk["fine_action_0"], True)]
                trajectory = trajectory + self.try_all_generator(temp_chunk["fine_action_1"])
                trajectory = trajectory + [(temp_chunk["fine_action_1"], True)]
                trajectory = trajectory + self.try_all_generator(temp_chunk["fine_action_2"])
                trajectory = trajectory + [(temp_chunk["fine_action_2"], True)]

            else:
                for action in temp_chunk['actions']:
                    trajectory = trajectory + self.try_all_generator(action)

        for action in trajectory:
            assert len(action[0]) == 8
        
        print("original action sum: {}".format(self.calculate_action_sum(episode_path)[:-1]))
        print("generated action sum: {}".format(self.calculate_generated_action_sum(trajectory)[:-1]))
        
        assert np.allclose(self.calculate_action_sum(episode_path)[:-1], self.calculate_generated_action_sum(trajectory)[:-1])
            
        return trajectory
        
    
def main(dir):
    
    action_aug = ActionAugmentation()
    
    parsed_traj = action_aug.calculate_action_values(dir)
    
    print(parsed_traj)
    print('\n')
    
    sampled_vectors = action_aug.trajectory_sampler(delta_coord = parsed_traj[0]["action_sum"], seed=0)
    print(sampled_vectors)
    print('\n')
    
    trajectory = action_aug.trajectory_generator(dir)
    print(trajectory)
    print('\n')
    
    sample_action = [0,0,1,0,0,1,1,1]
    print(action_aug.try_all_generator(sample_action))
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=episode_dir, help='path')
    parser.add_argument('--mode', type=str, default="save_file", help='path')

    args = parser.parse_args()
    if args.mode == "demo":
        task = 4
        episode = 0
        root_dir = "/home/jhkim/data/clipRT/rlds_dataset_builder/clip_rt_known_nonzeros/data/raw/"
        episode_dir = os.path.join(root_dir, "task_{}".format(task), "episode_{}".format(episode))
        main(args.dir)
    elif args.mode == "save_file":
        root_dir = "/home/jhkim/data/clipRT/rlds_dataset_builder/clip_rt_known_nonzeros/data/raw/"
        save_dir = "/home/jhkim/data/clipRT/rlds_dataset_builder/clip_rt_known_nonzeros/data/augmented/"
