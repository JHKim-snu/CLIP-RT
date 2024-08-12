# CLIP-RT
A foundation model for vision-language-action 

Clone the original openVLA git repo. 
To convert your own dataset to RLDS Dataset format (for fine-tuning openVLA)



## Table of Contents
* [OpenVLA](#OpenVLA)
* [Collect Training Data](#Collect-Training-Data)
* [Finally Testing on Your Robot](#Finally-Testing-on-Your-Robot)



## OpenVLA
### Construct Environment

Install the following dependencies.
```shell
conda create -n openvla python=3.10
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
cd openvla
pip install -e .
pip install packaging ninja
ninja --version; echo $?
pip install "flash-attn==2.5.5" --no-build-isolation

pip install accelerate
pip install opencv-python
pip install math3d
pip install pyk4a
pip install peft==0.11.1
pip install draccus
python -m pip install rich

```
If you have a problem with executing UR5, check the version of math3d. It may not be compatable.

### Inference
To test the OpenVLA model with a random test image and text, run
```shell
python test.py
```

### Finetune with LoRA
To finetune the openVLA model with your own data, you first need to preprocess your own dataset.

1. Clone https://github.com/kpertsch/rlds_dataset_builder
2. Env
```shell
conda env create -f environment_ubuntu.yml
conda activate rlds_env
```
3. Change `example_dataset` to `clip_rt_example`
4. Make a data directory `data/raw`
5. Convert the raw data to `.npy` via `raw_to_npy.ipynb`
6. in `clip_rt_example_dataset_builder.py`, modify the following line: (For myself: just copy and paste the clip_rt_example) BUT, change the name of the class!!!!
```shell
def _info(self) -> tfds.core.DatasetInfo:
def _split_generators(self, dl_manager: tfds.download.DownloadManager):
def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
```
Don't forget to edit the name of the python file "<dataset_name>_dataset_builder.py"
5. Then, execute
```shell
tfds build --overwrite
```

You will then get a RLDS format of your own dataset in `~/tensorflow_datasets/<name_of_your_dataset>`


In `openvla/prismatic/vla/datasets/rlds/oxe/configs.py`, include:

```shell
  "clip_rt_example": {
      "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
      "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
      "state_obs_keys": [None, None, None, None, None, None, None, None],
      "state_encoding": StateEncoding.NONE,
      "action_encoding": ActionEncoding.EEF_POS,
  },
```

Then, in `openvla/prismatic/vla/datasets/rlds/oxe/transforms.py`, include:
```shell
def clip_rt_example_dataset_transform():
...
```
and
```shell
OXE_STANDARDIZATION_TRANSFORMS = {
    "clip_rt_example": clip_rt_example_dataset_transform,
```

Then, run
```shell
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /home/jhkim/tensorflow_datasets \
  --dataset_name clip_rt_known \
  --run_root_dir /home/jhkim/data/clipRT/openvla/vla-scripts/runs \
  --adapter_tmp_dir /home/jhkim/data/clipRT/openvla/vla-scripts/adapter-tmp \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project cliprt 
```

With 4 H100 GPUs 100 steps is enough for training 10 episodes, taking less than 1 min.

After training:
unnorm key for my dataset seems not automatically added in config.json.
Manually copy it from `dataset_statics.json`. You MUST modify `pred_action` function in `openvla/prismatic/extern/hf/modeling_prismatic.py`, where the unrom_key is used. (WHAT TO DO? just copy the path of the `data_statistics.json` to `modeling_prismatic.py`.
Or you can try modifying `config.json`, by manually inserting the dataset statistics.

Test your model with:
```shell
python test.py --mode serveronly
```
---
### List of files that might be useful
- `openvla/prismatic/vla/datasets/rlds/dataset.py`
  ```shell
  def make_dataset_from_rlds
  ```

## Collect Training Data
### UR5
This is the code for collecting robotic action data with human lanugage.
- Robot Server
```shell
conda activate ur
```

Test the camera
```shell
python kinect_get_image.py
```

Test the robot
```shell
python ur_controller.py
```
You can check on the current joint pose by adding the argument `--getpose`.

To collect data, communicate with the remote server.
For the client (robot server), 
```shell
python client_ur_collect.py
```

## Finally Testing on Your Robot
### UR5
- Robot Server
```shell
conda activate ur
python client_ur_infer.py --mode safe
```
Set `--mode` to `cont` if you want for the robot to continuously act throughout the task.
For the safety mode, you must press the key for every action the robot receives. 
If you press "n", the robot does not take an action and returns to home pose.


- Remote Server (OpenVLA Server)
```shell
conda activate openvla
python test.py --mode full
```

