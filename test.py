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

parser = argparse.ArgumentParser()

parser.add_argument("--mode", dest="mode", action="store", default="full")
args = parser.parse_args()

vla_path = '/home/jhkim/data/clipRT/openvla/vla-scripts/runs/openvla-7b+clip_rt_known+b8+lr-0.0005+lora-r32+dropout-0.0'
adapter_path = "/home/jhkim/data/clipRT/openvla/vla-scripts/adapter-tmp/openvla-7b+clip_rt_known+b8+lr-0.0005+lora-r32+dropout-0.0"

unnorm_key = "clip_rt_known"


# Load Processor & VLA
# processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained(
#     "openvla/openvla-7b",
#     attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
# ).to("cuda")

processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    vla_path,
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda")

print("model loaded\n")
# vla = PeftModel.from_pretrained(vla, adapter_name=adapter_path)
# vla = AutoModelForVision2Seq.from_pretrained(adapter_path)

# vla.load_adapter(adapter_path)

vla.eval()


if args.mode == "serveronly":

    image = Image.open("/home/jhkim/data/clipRT/rlds_dataset_builder/clip_rt_example/data/pick_test/pick_test_0/0.jpg")

    instruction = "pick up the blue bottle"
    prompt = "In: What action should the robot take to {}?\nOut:".format(instruction)

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    print(action)



elif args.mode == "offline":

    while True:
        task_num = input("Insert task number (0 ~ 7) or 'x' to quit... ")
        if task_num == 'x':
            break
        episode_num = input("Insert episode number (0 ~ 9) ... ")
        episode_path = "/home/jhkim/data/clipRT/rlds_dataset_builder/clip_rt_known/data/raw/task_{}/episode_{}".format(task_num,episode_num)

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
        
        
elif args.mode == "full":
    # Execute...
    # robot.act(action, ...)

    image_root_path = "./raw_data/images/test/"
    json_root_path = "./raw_data/steps/test/"
    HOST = '127.0.0.1'
    HOST = '192.168.0.98'
    # HOST = '114.110.129.13'
    # HOST = '172.17.0.1'
    PORT = 9999

    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_sock.bind((HOST, PORT))
    srv_sock.listen()
    print("server listening...")
    cli_sock, addr = srv_sock.accept()
    print(f'Connected by: {addr}')

    image_cnt = 0
    instruction = ""
    
    while True:
        # try:
        data_save = {}
        cont = input("Press t for new task, otherwise, press any key to continue")
        if cont == 't':
            instruction = input("Instruction: ")
            
        # Receive CV2 Image
        received_data = cli_sock.recv(64).decode('utf-8')
        # print(received_data)
        # received_data = cli_sock.recv(65).decode('utf-8')
        # decoded_data = base64.b64decode(received_data)
        # length = int.from_bytes(decoded_data, byteorder='big')
        # print(f"Length: {length}")
        # length = received_data.strip()
        # length = ''.join(filter(lambda x: x in '01', length))
        # length = int(length)
        length = int(received_data)
        # print("Length sent:", length)
        
        buf = b''
        while length:
            newbuf = cli_sock.recv(length)
            buf += newbuf
            length -= len(newbuf)
        print("image recieved from the robot!")
        
        data = np.frombuffer(base64.b64decode(buf), np.uint8)
        cv2_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # save received image
        dt = datetime.datetime.now()
        save_image_path = image_root_path + '{}_{}_{}_{}_{}_{}.png'.format(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)
        cv2.imwrite(save_image_path, cv2_img)
        image_cnt += 1
        
        image = Image.open(save_image_path)
        image = image.resize((256, 256))
                    
        prompt = "In: What action should the robot take to {}?\nOut:".format(instruction)
        
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        
        
        action_info = f'{action[0]};{action[1]};{action[2]};{action[3]};{action[4]};{action[5]};{action[6]}'
        print(f'Send {action_info}')
        print('\n')
        cli_sock.send(action_info.encode())

        data_save['image_path'] = save_image_path
        data_save['instruction'] = instruction
        data_save['action'] = action_info
        
        json_path = os.path.join(json_root_path,'{}_{}_{}_{}_{}_{}.json'.format(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second))
        with open(json_path, 'w') as f:
            json.dump(data_save,f)
            ###############AAAAAAAAAAAAAAAAAA################
            #############################


        # except KeyboardInterrupt:
        #     print('\n Server Ctrl-c')
        #     break
        # except ValueError:
        #     print('\n Client Closed')
        #     break

    cli_sock.close()
    srv_sock.close()
