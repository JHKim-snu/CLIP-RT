# CLIP-RT
A foundation model for vision-language-action 
## OpenVLA
```shell
conda create -n openvla python=3.10
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
pip install flash-attn --no-build-isolation
pip install opencv-python
pip install math3d
pip install pyk4a
```
