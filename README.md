# CLIP-RT
A foundation model for vision-language-action 

## OpenVLA
Install the following dependencies.
```shell
conda create -n openvla python=3.10
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
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
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>
```
