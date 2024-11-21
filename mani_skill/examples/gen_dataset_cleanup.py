import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose
from utils_trajectory import DummyCamera, generate_curve_torch
from utils_traj_tokens import encode_trajectory_xy, encode_trajectory_xyz

from pdb import set_trace

#check that all files are there
from scipy.spatial.transform import Rotation as R
def to_rot(q):
        w, x, y, z = q
        return R.from_quat([x, y, z, w]).as_euler('xyz', degrees=True)

#LIMIT_LINES = 5000
LIMIT_LINES = None
def split_dataset(dataset_path: Path, train_ratio: float = 0.8, seed: int = 42):
    """
    Randomly split a JSONL dataset into train and test sets.

    Args:
        dataset_path (Path): Path to the JSONL file.
        train_ratio (float): The ratio of the data to use for training.
        seed (int): Random seed for reproducibility.
    """

    # Define paths for train and test files
    json_file = dataset_path / "_annotations.all.jsonl"
    train_path = dataset_path / "_annotations.train.jsonl"
    valid_path = dataset_path / "_annotations.valid.jsonl"

    # Load the dataset
    with open(json_file, "r") as file:
        lines = file.readlines()

    depths = []
    rots = []
    new_lines = []
    refered_files = []    
    for i, line in tqdm(enumerate(lines[:LIMIT_LINES]), total=len(lines)):
        try:
            label = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f"Error decoding line {i}: {line[:50]}")
            continue
        if "<loc1024>" in label["suffix"]:
            print("found loc1024 in", label["suffix"])
            continue
        
        refered_files.append(label["image"])

        camera = DummyCamera(label["camera_intrinsic"], label["camera_extrinsic"])
        start_pose = Pose(raw_pose=torch.tensor(label["start_pose"]))
        end_pose = Pose(raw_pose=torch.tensor(label["end_pose"]))
        _, curve_3d = generate_curve_torch(start_pose.get_p(), end_pose.get_p(), num_points=2)
        curve_2d, depth, curve_tokens = encode_trajectory_xyz(curve_3d, camera)

        new_label = dict(image=label["image"], prefix=label["prefix"], suffix=curve_tokens)
        new_line = json.dumps(new_label) + "\n"
        new_lines.append(new_line)


        # from utils_traj_tokens import parse_trajectory_xyz
        # curve_2d_ref, depth_ref = parse_trajectory_xyz(curve_tokens, camera)
        # assert torch.allclose(curve_2d, curve_2d_ref.unsqueeze(0), atol=0.6) # 0.6 pixel
        # assert torch.allclose(depth, depth_ref.unsqueeze(0), atol=0.006) # 0.6 cm

        # # verify old trajectory
        # _, curve_3d_ref = generate_curve_torch(start_pose.get_p(), end_pose.get_p(), num_points=20)
        # traj2d_ref = encode_trajectory_xy(curve_3d_ref, camera)
        # if traj2d_ref != label["suffix"]:
        #     print(f"mismatch in line {i}: ", label["image"])
        
        # depths.append(depth)
        # rots.append(to_rot(label["traj_q"][0]))

    #rots = np.array(rots)
    #import matplotlib.pyplot as plt
    #plt.hist(rots[:, 2])
    #tmp = torch.cat(depths).flatten().tolist()
    #plt.hist(tmp)
    
    lines = new_lines
    set_trace()

    print("loading all .jpg images in", dataset_path)
    present_files = list(dataset_path.glob("*.jpg"))
    refered_files_set = set(refered_files)
    present_files_set = {file.name for file in present_files}
    missing_files = refered_files_set - present_files_set
    extra_files = present_files_set - refered_files_set
    print("referred but not present:", len(missing_files))
    print("present but not referred:", len(extra_files))
    assert len(missing_files) == 0

    # Set the random seed for reproducibility
    random.seed(seed)
    # Shuffle and split the dataset
    random.shuffle(lines)
    split_index = int(len(lines) * train_ratio)
    train_lines = lines[:split_index]
    valid_lines = lines[split_index:]

    # Write to the train and test files
    with open(train_path, "w") as train_file:
        train_file.writelines(train_lines)
    with open(valid_path, "w") as valid_file:
        valid_file.writelines(valid_lines)
    print("train images", len(train_lines))
    print("valid images", len(valid_lines))
    
    create_zip = False
    if create_zip:
        # create a zip of dataset_path.parent and place it in dataset_path.parent.parent
        print("\nCreating zip, this may take a while...")
        print(dataset_path.parent)
        source_dir = dataset_path.parent
        destination_dir = dataset_path.parent.parent / f"{source_dir.name}.zip"
        zip_path = shutil.make_archive(base_name=destination_dir.with_suffix(''), format='zip', root_dir=source_dir)
        print(f"Created zip at {zip_path} with size: {Path(zip_path).stat().st_size / (1024 * 1024):.2f} MB and {len(lines)} samples")

    print("done.")

if __name__ == "__main__":
    dataset_path = Path("/tmp/clevr-act-2/dataset")
    split_dataset(dataset_path)