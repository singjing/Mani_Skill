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
