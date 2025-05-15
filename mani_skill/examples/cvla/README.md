# cVLA Environment

This is the code for the cVLA env. This is an environment that generates language conditioned tasks in a modular manner. It is modular with respect to several components,
that can be specified using command line arguments for the `run_env.py` script.
1. Object datasets ("-od")
2. Robots ("-r")
3. Scenes ("-sd" and "-so")
4. Camera randomizations ("-cv")
5. Actions (arguments not implemented)

While it can be used as a general VLA setup, there are currently some differences to standard VLA setups.
1. One-step actions: a full trajectory is predicted from the initial observation
2. Camera-space actions: actions can be predicted in image frame coordinates
3. Depth input: the setup is designed to allow for models to use depth input

This code lets you easily run the environment for debugging purposes, generate data, and evaluate policies in simulation. In the following sections, we will describe each of these. An example of training code can be found in the [cVLA repo](https://github.com/BlGene/cVLA).

## Setup
Setup is only needed for generating data and using objaverse assets. Go to the `cvla_paths.py` file and enter the required information.

## Running the Environment
Create a conda env (I'm currently using python 3.12)
```bash
conda create env -n maniskill
conda activate maniskill
cd ManiSkill/mani_skill/examples
python run_env.py  # runs the scripted policy in viewer (requires display support).
```

The environment can be run in a number of different modes, these are specified by ("-m") and they are:
1. **script:** use the built-in solver to solve the task (default).
2. **interactive:** interactive viewer
3. **first:** just generate the first frame (used for dataset generation)


## Generating Simulation Data
Datasets can be generated using the `run_env.py` scrip, this is done by specifying a
`--record_dir` to save the data to. By default, this will turn off the visualization window. Generating scenes should be deterministic, this can be achieved by setting seed(s). An example is provided below. The `dataset_merge_h5.py` scipts merges invidual `.h5` files into
one large one, which can be loaded using a data loader.

```bash
ssh <compute-node>  # can be very small
conda activate maniskill
source .bashrc
cd ManiSkill/mani_skill/examples

CUDA_VISIBLE_DEVICES=0 python run_env.py -cv random_fov -so random --record_dir /tmp/cvla-clevr-9-camRF-sceneR --N_samples=150000
CUDA_VISIBLE_DEVICES=1 python run_env.py -cv random_fov -so random -od objaverse --record_dir /tmp/cvla-obja-9-camRF-sceneR --N_samples=75000
python dataset_creation/dataset_merge_h5.py --root_dir /tmp/cvla-clevr-9-camRF-sceneR
python dataset_creation/dataset_merge_h5.py --root_dir /tmp/cvla-obja-9-camRF-sceneR 
```

For creating several datasets, see `slurm_create_datasets.sh`


## Evaluating in Simulation
For the training code, as well as a more practical implementaion of an evaluation, please see the [cVLA repo](https://github.com/BlGene/cVLA). This is just a short sketch for an example simulation evaluation.

```python
from mani_skill.examples.run_env import Args, iterate_env

parsed_args = Args()
parsed_args.env_id = "ClevrMove-v1"
parsed_args.render_mode = "rgb_array"
parsed_args.control_mode = "pd_joint_pos"
parsed_args.camera_views = "random_side"
parsed_args.object_dataset = "objaverse"
parsed_args.action_encoder = "xyzrotvec-cam-512xy128d"
env_iter = iterate_env(parsed_args, vis=False, model=model_wrapped)

reward_succes = 0
for i in range(50):
  json_dict = next(env_iter)[1]
  if json_dict["reward"] > 0.75:
    reward_succes += 1
```

## Modular Components
Here, we describe briefly each of the modular components of this environment. Most of these are settable using command line flags for `run_env.py`.


### Object Datasets
The object dataset can be set using the command line flag "-od". Current options are clevr, ycb, objaverse.
1. **clevr:** a set of shapes inspired by clevr, currently cubes, spheres, and blocks in 8 colors and 2 sizes.
2. **ycb:** YCB objects, the standard maniskill YCB objects (I don't use these.)
3. **objaverse:** objaverse shapes downloaded from objaverse.

If you want to modify this behaviour, look at the `_load_scene_clevr`, `_load_scene_ycb`, and `_load_scene_objaverse` files in `cvla_env.py`.


#### Objaverse Assets
Getting from the raw objaverse assets to something usable for simulation is a bit involved. There is a lot of filtering, e.g. by size and descriptions. The data will be made avaliable for download. The code for the process of generating this data is in the `cvla-dataset-creation` repo.


### Robots
The robot can be set using the command line flag "-r". Various ManiSkill robots are supported, currently I have used: panda, panda_wristcam, xarm6_robotiq, floating_inspire_hand_right, fetch.

The current demonstration generation and policy execution code requires IK control and this not available for all robots, e.g. the fetch robot, so please be aware of this potential constraint.


### Scenes

Scene setup is divided into two parts, scene datasets and scene options.
The scene dataset can be selected using the command line flag "-sd". Current options are
1. **Table:** 4 different table configurations, with color randomization. (default)
2. **ProcTHOR:** ProcTHOR houses, currently I've only tested one of the scenes.

Scene options can be set using the "-so" flag. Current options are:
1. **fixed:** Always load the same variant of a scene. (default)
2. **random:** Randomize the scene, according to it's `NUM_SCENE_OPTIONS` variable.

If you want to modify this behavior, look at the `_load_scene` function in `cvla_env.py`.


### Camera View Randomization
The camera randomization can be set using the command line flag "-cv". The current options are
1. **fixed:** fixed perspective.
2. **random_side:** randomized perspective, but looking from the side.
3. **random:** randomized perspective also from top.
4. **random_fov:** randomized perspective also from top, fov randomization.

If you want to modify this behavior, look at the `initalize_render_camera` function in `cvla_env.py`.


### Actions
This is WIP, no command line option yet. Currently, only `moveobject_onto` is used. There is code for:

1. move_object_onto: move A onto B (default)
2. move_object_next_to
3. move_object_leftrightbehind
4. move_object_between
5. move_object_rotate_x
6. move_object_lr_singular
7. move_object_forward_backward


## Misc Notes

## Running other Environments
```bash
python -m mani_skill.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
 --render-mode="human"
```