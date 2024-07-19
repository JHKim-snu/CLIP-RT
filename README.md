# CLIP-RT
A foundation model for vision-language-action 

## OpenVLA
```shell
conda create -n openvla python=3.10
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
pip install accelerate
pip install flash-attn --no-build-isolation
pip install opencv-python
pip install math3d
pip install pyk4a
pip install peft==0.11.1
```
If you have a problem with executing UR5, check the version of math3d. It may not be compatable.

To test the OpenVLA model with a random test image and text, run
```shell
python test.py
```

To finetune the openVLA model with your own data, run
```shell
torch
```
