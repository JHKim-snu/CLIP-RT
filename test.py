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

# ROBOT
# import urx
# from ur_controller import RobotController
# rc = RobotController()
# Grab image input & format prompt
# capture = rc.camera.get_capture()
# cv_img = capture.color[:, :, :3]
# image = cv_img


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

vla_path = '/home/jhkim/data/clipRT/openvla/vla-scripts/runs/openvla-7b+clip_rt_example+b8+lr-0.0005+lora-r32+dropout-0.0'
adapter_path = "/home/jhkim/data/clipRT/openvla/vla-scripts/adapter-tmp/openvla-7b+clip_rt_example+b8+lr-0.0005+lora-r32+dropout-0.0"

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


# vla = PeftModel.from_pretrained(vla, adapter_name=adapter_path)
# vla = AutoModelForVision2Seq.from_pretrained(adapter_path)

# vla.load_adapter(adapter_path)

# vla.eval()


if args.mode == "serveronly":

    image = Image.open("/home/jhkim/data/clipRT/rlds_dataset_builder/clip_rt_example/data/pick_test/pick_test_0/0.jpg")

    instruction = "pick up the blue bottle"
    prompt = "In: What action should the robot take to {}?\nOut:".format(instruction)

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="clip_rt", do_sample=False)

    print(action)


elif args.mode == "full":
    # Execute...
    # robot.act(action, ...)

    image_root_path = "./raw_data/images/test/"
    json_root_path = "./raw_data/steps/test/"
    HOST = '114.110.129.13'
    PORT = 9998

    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_sock.bind((HOST, PORT))
    srv_sock.listen()
    cli_sock, addr = srv_sock.accept()
    print(f'Connected by: {addr}')

    image_cnt = 0
    instruction = ""
    
    while True:
        try:
            data = {}
            cont = input("Press "t" for new task, otherwise, press any key to continue")
            if cont == 't':
                instruction = input("Instruction: ")
                
            # Receive CV2 Image
            length = int(cli_sock.recv(65).decode('utf-8'), 2)
            buf = b''
            while length:
                newbuf = cli_sock.recv(length)
                buf += newbuf
                length -= len(newbuf)
            print("image recieved from the robot!")
            
            data = np.frombuffer(base64.b64decode(buf), np.uint8)
            cv2_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            # save received image
            dt = datetime.now()
            save_image_path = image_root_path + '{}_{}_{}_{}_{}_{}.png'.format(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)
            cv2.imwrite(save_image_path, cv2_img)
            image_cnt += 1
            
            image = Image.open(save_image_path)
                        
            prompt = "In: What action should the robot take to {}?\nOut:".format(instruction)
            
            inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            action = vla.predict_action(**inputs, unnorm_key="clip_rt", do_sample=False)
            
            
            action_info = f'{action[0]};{action[1]};{action[2]};{action[3]};{action[4]};{action[5]};{action[6]}'
            print(f'Send {action_info}')
            print('\n')
            cli_sock.send(action_info.encode())

            data['image_path'] = save_image_path
            data['instruction'] = instruction
            data['action'] = action
            
            json_path = os.path.join(json_root_path,'{}_{}_{}_{}_{}_{}.json'.format(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second))
            with open(json_path, 'w') as f:
                json.dump(data,f)
            ###############AAAAAAAAAAAAAAAAAA################
            #############################


        except KeyboardInterrupt:
            print('\n Server Ctrl-c')
            break
        except ValueError:
            print('\n Client Closed')
            break

    cli_sock.close()
    srv_sock.close()
