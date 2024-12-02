from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

import os
import cv2
import base64
import socket
import numpy as np
import time
from peft import PeftModel 

import socket
import base64
import argparse
import datetime
import json
import math

parser = argparse.ArgumentParser()

parser.add_argument("--mode", dest="mode", action="store", default="online")
parser.add_argument("--data", action="store", default="known")
parser.add_argument("--task", action="store", default="0")
parser.add_argument("--episode", action="store", default="0")
parser.add_argument("--step", action="store", default="0")
parser.add_argument("--debug", action="store", default="no")
parser.add_argument("--model", dest="model", action="store", default="cliprt")
args = parser.parse_args()


if args.model == 'openvla':
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")
    
else:
    # name of the dataset that the model is trained on
    dataset_name = "clip_rt_expert"

    vla_path = '/home/jhkim/data/clipRT/openvla/vla-scripts/runs/openvla-7b+{}+b8+lr-0.0005+lora-r32+dropout-0.0'.format(dataset_name)
    adapter_path = "/home/jhkim/data/clipRT/openvla/vla-scripts/adapter-tmp/openvla-7b+{}+b8+lr-0.0005+lora-r32+dropout-0.0".format(dataset_name)
    unnorm_key = dataset_name

    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")

print("model loaded\n")
vla.eval()


########################################################################
# This MODE is to test the model on a single image
########################################################################
if args.mode == "single":

    image = Image.open("test1.png")

    instruction = "pick up the blue bottle"
    prompt = "In: What action should the robot take to {}?\nOut:".format(instruction)

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    print(action)
                
                
########################################################################
# This MODE is to test the model on images of a specific episode. 
########################################################################
elif args.mode == "offline":

    while True:
        task_num = input("Insert task number (0 ~ 7) or 'x' to quit... ")
        if task_num == 'x':
            break
        episode_num = input("Insert episode number (0 ~ 9) ... ")
        episode_path = "/home/jhkim/data/clipRT/rlds_dataset_builder/{}/data/raw/task_{}/episode_{}".format(dataset_name,task_num,episode_num)

        data_files = os.listdir(episode_path)
        
        for file in data_files:
            data_file_path = os.path.join(episode_path,file)
            if '.json' in data_file_path:
                with open(data_file_path,'r') as f:
                    data = json.load(f)
                image_path = os.path.join(episode_path, data['image_path'].split("/")[-1])
                with Image.open(image_path) as img:
                    image = img.resize((256,256))
                    
            instruction = data['instruction']
            prompt = "In: What action should the robot take to {}?\nOut:".format(instruction)

            # Predict Action (7-DoF; un-normalize for BridgeData V2)
            inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

            print(action)
            
            input("Press any Key to continue ... ")
        
        print("\nEpisode Done.")
        

########################################################################
# This MODE is to test the model on a physical robot
########################################################################
elif args.mode == "online":
    print("ONLINE EXPERIMENT")

    image_root_path = "./inference_data/images/test/"
    json_root_path = "./inference_data/steps/test/" # path to save action data
    HOST = '' # IP address of HOST server
    PORT = 9998

    instruction = input("Instruction: ")
    
    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_sock.bind((HOST, PORT))
    srv_sock.listen()
    print("server listening...")
    cli_sock, addr = srv_sock.accept()
    print(f'Connected by: {addr}')

    image_cnt = 0
    
    crop_ = 256  # We crop the left 256 pixels of the image
    
    while True:
        data_save = {}
        
        # Receive CV2 Image
        received_data = cli_sock.recv(64).decode('utf-8')
        length = int(received_data)        
        buf = b''
        while length:
            newbuf = cli_sock.recv(length)
            buf += newbuf
            length -= len(newbuf)
        print("image recieved from the robot!")
        
        # save received image
        data = np.frombuffer(base64.b64decode(buf), np.uint8)
        cv2_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        cv2_img = cv2_img[:, crop_:]
        dt = datetime.datetime.now()
        save_image_path = image_root_path + '{}_{:02}_{:02}_{:02}_{:02}_{:02}.png'.format(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)
        cv2.imwrite(save_image_path, cv2_img)
        image_cnt += 1
        
        image = Image.open(save_image_path)
        image = image.resize((256, 256))
                    
        prompt = "In: What action should the robot take to {}?\nOut:".format(instruction)
        
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        print(action)
        
        action_info = f'{action[0]};{action[1]};{action[2]};{action[3]};{action[4]};{action[5]};{action[6]};{instruction}'
        print(f'Send {action_info}')
        print('\n')
        cli_sock.send(action_info.encode())

        data_save['image_path'] = save_image_path
        data_save['instruction'] = instruction
        data_save['action'] = action_info
        
        json_path = os.path.join(json_root_path,'{}_{}_{}_{}_{}_{}.json'.format(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second))
        with open(json_path, 'w') as f:
            json.dump(data_save,f)

    cli_sock.close()
    srv_sock.close()
    
    
########################################################################
# This MODE is to test the model on the training set. 
# Test the accuracy of the predicted action on the training set.
########################################################################
if args.mode == "getscore":
    correct = 0
    correct_dot = 0
    correct_xyz = 0
    correct_rpy = 0
    correct_grip = 0
    wrong = 0
    wrong_dot = 0
    wrong_xyz = 0
    wrong_rpy = 0
    wrong_grip = 0
    sample_num = 0
    execution_time = 0
    
    raw_root = "" # Path of the dataset to test on, /rlds_dataset_builder/clip_rt_known_augmented/data/raw/train_postprocessed
    tasks = [f for f in os.listdir(raw_root) if not f.startswith('.')]

    for task in tasks:
        task_path = os.path.join(raw_root, task)
        episodes = [f for f in os.listdir(task_path) if not f.startswith('.')]
        for episode in episodes:
            if episode == "episode_0":
                episode_path = os.path.join(task_path, episode)
                img_files = [f for f in os.listdir(episode_path) if f.endswith('.png')]
                for img_file in img_files:
                    test_image_path = os.path.join(episode_path, img_file)
                    
                    test_json_path = test_image_path.split(".png")[0] + ".json"
                    
                    image = Image.open(test_image_path)

                    with open(test_json_path, 'r') as file:
                        data = json.load(file)
                    
                    start_time = time.time()
                    
                    instruction = data["instruction"]
                    supervision = data["supervision"]
                    gt_action = data["action"]
                    
                    prompt = "In: What action should the robot take to {}?\nOut:".format(instruction)

                    # Predict Action (7-DoF; un-normalize for BridgeData V2)
                    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
                    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
                    
                    end_time = time.time()
                    
                    execution_time += end_time - start_time
                    sample_num += 1
                    
                    gt_action_simple = [None,None,None,None,None,None,None]
                    for i,e in enumerate(gt_action[0:3]):
                        if e>0.005:
                            gt_action_simple[i] = 1
                        elif e<-0.005:
                            gt_action_simple[i] = -1
                        else:
                            gt_action_simple[i] = 0
                    for i,e in enumerate(gt_action[3:6]):
                        degree = e*180/math.pi
                        if degree>10:
                            gt_action_simple[i+3] = 1
                        elif degree<-10:
                            gt_action_simple[i+3] = -1
                        else:
                            gt_action_simple[i+3] = 0
                    if gt_action[-1] > 0.5:
                        gt_action_simple[-1] = 1
                    else:
                        gt_action_simple[-1] = 0

                    action_simple = [None,None,None,None,None,None,None]
                    for i,e in enumerate(action[0:3]):
                        if e>0.005:
                            action_simple[i] = 1
                        elif e<-0.005:
                            action_simple[i] = -1
                        else:
                            action_simple[i] = 0
                    for i,e in enumerate(action[3:6]):
                        degree = e*180/math.pi
                        if degree>10:
                            action_simple[i+3] = 1
                        elif degree<-10:
                            action_simple[i+3] = -1
                        else:
                            action_simple[i+3] = 0
                    if action[-1] > 0.5:
                        action_simple[-1] = 1
                    else:
                        action_simple[-1] = 0
                    
                    if gt_action_simple == action_simple:
                        correct += 1
                    else:
                        wrong += 1
                        
                    if gt_action_simple[0:3] == action_simple[0:3]:
                        correct_xyz += 1
                    else:
                        wrong_xyz += 1
                        
                    if gt_action_simple[3:6] == action_simple[3:6]:
                        correct_rpy += 1
                    else:
                        wrong_rpy += 1
                        
                    if gt_action_simple[6] == action_simple[6]:
                        correct_grip += 1
                    else:
                        wrong_grip += 1
                        
                    if np.dot(gt_action_simple[:-1], action_simple[:-1])>0:
                        correct_dot += 1
                    else:
                        wrong_dot += 1
                        
                    if (correct + wrong) % 50 == 0: 
                        print("{} sampled processed ... ".format(correct+wrong))

    print("Number of samples: {}".format(correct+wrong))
    print("Accuracy: {}".format(100*correct/(correct+wrong)))
    print("Dot Accuracy: {}".format(100*correct_dot/(correct_dot+wrong_dot)))
    print("XYZ Accuracy: {}".format(100*correct_xyz/(correct_xyz+wrong_xyz)))
    print("RPY Accuracy: {}".format(100*correct_rpy/(correct_rpy+wrong_rpy)))
    print("Gripper Accuracy: {}".format(100*correct_grip/(correct_grip+wrong_grip)))
    
    print("EXECUTION TIME: {}".format(execution_time))
    print("number of samples: {}".format(sample_num))
    print("AVERAGE EXECUTION TIME: {}".format(execution_time/sample_num))