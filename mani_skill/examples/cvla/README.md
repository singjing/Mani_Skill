# cVLA Dataset Creation



## Generate the dataset
Create a conda env (I'm currently using python 3.12)
```
conda create env -n maniskill
conda activate maniskill
cd ManiSkill/mani_skill/examples
python run_env.py  # runs the scripted policy in viewer (requires display support)
```


# Generating Simulation Data

```
ssh <compute-node>  # can be very small
conda activate maniskill
source .bashrc
cd ManiSkill/mani_skill/examples

CUDA_VISIBLE_DEVICES=2 python run_env.py -cv random_fov -so random --record_dir /tmp/cvla-clevr-9-camRF-sceneR --N_samples=150000
CUDA_VISIBLE_DEVICES=3 python run_env.py -cv random_fov -so random -od objaverse --record_dir /tmp/cvla-obja-9-camRF-sceneR --N_samples=75000
```

For creating a bunch of dataset see `slurm_create_datasets.sh`


## Other simultation envs
```
python -m mani_skill.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="human"
```
