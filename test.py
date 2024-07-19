from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

import os
import cv2
import base64
import socket
import numpy as np
import time

# ROBOT
# import urx
# from ur_controller import RobotController
# rc = RobotController()
# Grab image input & format prompt
# capture = rc.camera.get_capture()
# cv_img = capture.color[:, :, :3]
# image = cv_img


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda")

image = Image.open("../RT/azure_test.png")

instruction = "pick the cup"
prompt = "In: What action should the robot take to {}?\nOut:".format(instruction)

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(action)
# Execute...
# robot.act(action, ...)
