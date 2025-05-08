# cVLA Dataset Creation

## Generate the dataset
Create a conda env (I am using 3.10, because of some maniskill/tensorflow issues I think.)
```
conda create env -n maniskill
conda activate maniskill

cd ManiSkill/mani_skill/examples
python run_env.py  # see interactive viewer
python run_env.py --record_dir /tmp/clevr-dataset  # generate dataset
rsync -a --progress /tmp/clevr-dataset /data/lmbraid19/argusm/datasets  # copy to server
```

## Other simultation envs
```
python -m mani_skill.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="human"
```


# Generating Simulation Data

```
conda activate paligemma
CUDA_VISIBLE_DEVICES=X python run_env.py --record_dir /tmp/cvla-clevr-8 --N_samples=150000
CUDA_VISIBLE_DEVICES=X python run_env.py --record_dir /tmp/cvla-obja-8  -od objaverse --N_samples=75000
```

For creating a bunch of dataset see `slurm_create_datasets.sh`


ssh chap
conda activate paligemma
source .bashrc
cd /ihome/argusm/lang/ManiSkill/mani_skill/examples/

CUDA_VISIBLE_DEVICES=2 python run_env.py -cv random_fov -so random --record_dir /tmp/cvla-clevr-9-camRF-sceneR --N_samples=150000
CUDA_VISIBLE_DEVICES=3 python run_env.py -cv random_fov -so random -od objaverse --record_dir /tmp/cvla-obja-9-camRF-sceneR --N_samples=75000