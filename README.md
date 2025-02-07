<div align="center">
  
<h1>CLIP-RT</h1>
  
**[Gi-Cheon Kang][3]<sup>\*</sup>, &nbsp; [Junghyun Kim][4]<sup>\*</sup>, &nbsp; [Kyuhwan Shim][5], &nbsp; [Jun Ki Lee][6], &nbsp; [Byoung-Tak Zhang][1]** <br>

**[3rd Workshop on Language and Robot Learning @ CoRL2024][2]**
</div>

<h3 align="center">
<a href="https://arxiv.org/abs/2411.00508">arXiv</a> | <a href="https://clip-rt.github.io/">Project Page</a> 
</h3>

### 
<img src="readme_figure/cliprt_overview.gif" width="100%" align="middle"><br><br>


This repository provides detailed instructions on:

1. Training [OpenVLA][0]
2. Collecting data using natural language supervision
3. Conducting experiments with [UR5][8]

Main CLIP-RT model can be found on [this repo][7].



## Table of Contents
* [OpenVLA](#OpenVLA)
* [Collect Training Data](#Collect-Training-Data)
* [Finally Testing on Your Robot](#Finally-Testing-on-Your-Robot)



## OpenVLA

### Prepare Dataset
To finetune the openVLA model with your own data, you first need to convert your own dataset to RLDS Dataset format.

1. Clone https://github.com/kpertsch/rlds_dataset_builder
2. Environment
```shell
conda env create -f environment_ubuntu.yml
conda activate rlds_env
```
3. Change directory name `example_dataset` to `[your_dataset_name]`
4. Make a data directory `[your_dataset_name]/data/raw`

We expect data to be uploaded to the following directory structure:

    ├── raw         
    │   ├── task_0       
    │   │   ├── episode_0
    │   │   │   ├── x.png      
    │   │   │   ├── x.json      
    │   │   │   └── ...     
    │   │   ├── episode_1   
    │   │   │   ├── y.png   
    │   │   │   └── ...      
    └── 

Each elements in `.json`file consists of the initial instruction of the task, natural language supervision, low-level action vector (7-d and 8-d), file name of the image as shown below.

<pre>
{"prev_eef_pose": [0.4867851071573382, 0.11065651089581363, 0.43149384252823314, 3.1369514868261072, 0.006225836816466727, 0.0016994614054684717], "prev_joint": [3.1415085792541504, -1.5707486311541956, 1.570798397064209, 4.71236515045166, -1.5706971327411097, -1.5709307829486292], "eef_pose": [0.48678753690778354, 0.210619672338698, 0.43147651408980553, 3.136830345592423, 0.006070714275545524, 0.0016729409424801845], "joint": [3.3396871089935303, -1.4953535238849085, 1.4921374320983887, 4.716606140136719, -1.570972744618551, -1.3726237455951136], "instruction": "open the cabinet", "supervision": "move arm to the left", "action": [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "image_path": "2024_9_11_13_38_56.png", "openx_action": [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0]}
</pre>


5. We provide functions to preprocess the raw data. Extracting 7-d action from 8-d action for example. `data_utils/preprocess_raw.py`
6. Convert the raw data to `.npy` format via `data_utils/convert_to_RLDS.py`
7. In `[your_dataset_name]_dataset_builder.py`, modify the following lines according to your dataset:
```shell
def _info(self) -> tfds.core.DatasetInfo:
def _split_generators(self, dl_manager: tfds.download.DownloadManager):
def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
```
Modify the name of the class as following rule:
`my_data` to `class MyData():`

8. Finally, execute
```shell
tfds build --overwrite
```

You will then get a RLDS format of your own dataset in `~/tensorflow_datasets/<name_of_your_dataset>`



### Finetune with LoRA

Clone the [OpenVLA git repo][9]. 

Install the following dependencies.
```shell
conda create -n openvla python=3.10
pip install -r requirements.txt
cd openvla
pip install -e .
```

To test the OpenVLA model with a random test image and text, run
```shell
python test.py --mode single --model openvla
```

To fine-tune the model:

In `openvla/prismatic/vla/datasets/rlds/oxe/configs.py`, include:

```shell
  "[name_of_your_dataset]": {
      "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
      "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
      "state_obs_keys": [None, None, None, None, None, None, None, None],
      "state_encoding": StateEncoding.NONE,
      "action_encoding": ActionEncoding.EEF_POS,
  },
```

Then, in `openvla/prismatic/vla/datasets/rlds/oxe/transforms.py`, include:
```shell
def [name_of_your_dataset]_dataset_transform():
...
```
and
```shell
OXE_STANDARDIZATION_TRANSFORMS = {
    "[name_of_your_dataset]": [name_of_your_dataset]_dataset_transform,
```

Then, run
```shell
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir ~/tensorflow_datasets \
  --dataset_name [name_of_your_dataset] \
  --run_root_dir /openvla/vla-scripts/runs \
  --adapter_tmp_dir /openvla/vla-scripts/adapter-tmp \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project [name] 
```

With 4 H100 GPUs 100 steps is enough for training 10 episodes, taking only couple of min.


### Test Your Model

Unnorm key seems not automatically added in config.json.
Manually copy it from `dataset_statics.json` under `vla-scripts/runs`. 
You MUST modify `pred_action` function in `openvla/prismatic/extern/hf/modeling_prismatic.py`, where the unrom_key is used OR you can try modifying `config.json`, by manually inserting the dataset statistics.

Now, you're ready to test your model with:
```shell
python test.py --mode serveronly
```
---


## Collect Training Data
### Robot Server (Client)
This is the code for collecting robotic action data with UR5.
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

### Remote Server
Comming Soon!

### Stochastic Trajectory Diversification
Comming Soon!


## CLIP-RT Dataset

The `common.zip` and `novel.zip` files each contain datasets for multiple tasks (9 common tasks and 10 novel tasks).
These datasets were collected using human natural language supervisions. For each task, we collected 10 episodes.

Each episode consists of a series of PNG-JSON pairs:

1. PNG: An image representing the current scene.
2. JSON: Metadata including the initial human instruction, natural language supervision, low-level actions, and other relevant details.



| Name  | Content | Examples | Size | Link |
| --- | --- |--- | --- |--- |
| `common.zip`  | Data for common tasks collected through natural language supervision | 911 | 670 MBytes | [Download](https://drive.google.com/drive/folders/12fHThp8IC1fzmbyx850zSkcuFXUQsy9V?usp=sharing)|
| `common_augmented.zip`  | Augmented data of common tasks | 9,841 | 1.8 GBytes | [Download](https://drive.google.com/drive/folders/1QpsvN-y-MJtq6r9tqoubWFPXwl6iuskA?usp=sharing)|
| `novel.zip`  | Data for novel tasks collected through natural language supervision | 1,276 | 1.2 GBytes | [Download](https://drive.google.com/drive/folders/12fHThp8IC1fzmbyx850zSkcuFXUQsy9V?usp=sharing)|
| `novel_augmented.zip`  |  Augmented data of novel tasks | 11,578 | 2.0 GBytes | [Download](https://drive.google.com/drive/folders/1-7K7Nv0n6ax5lVWEYWYQjrdzpsEjhRdp?usp=sharing)|


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


- Remote Server

(OpenVLA)
```shell
conda activate openvla
python test.py --mode full

```
(CLIP-RT)
Check out [this repo][7].

[0]: https://openvla.github.io/
[1]: https://bi.snu.ac.kr/~btzhang/
[2]: https://sites.google.com/view/langrob-corl24/home?authuser=0
[3]: https://gicheonkang.com
[4]: https://jhkim-snu.github.io/
[5]: https://underthelights.github.io/
[6]: https://junkilee.github.io/
[7]: https://github.com/gicheonkang/clip-rt
[8]: https://www.universal-robots.com/products/ur5-robot/
[9]: https://github.com/openvla/openvla
