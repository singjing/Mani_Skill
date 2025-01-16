import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose
from utils_trajectory import DummyCamera, generate_curve_torch
from utils_traj_tokens import encode_trajectory_xyz

from pdb import set_trace

#check that all files are there
from scipy.spatial.transform import Rotation as R
def to_rot(q):
        w, x, y, z = q
        return R.from_quat([x, y, z, w]).as_euler('xyz', degrees=True)

def check_and_fix_format(label):
    if "<loc1024>" in label["suffix"]:
        print("found loc1024 in", label["suffix"])
    if "<" in label["prefix"]:
        index = label["prefix"].index("<")
        if index != 0 and label["prefix"][index-1] != " ":
            label["prefix"] = label["prefix"].replace("<", " <", 1)
    if '<loc-' in label["suffix"]:
        print("found <loc- in", label["suffix"])
    if '<loc-' in label["prefix"]:
        print("found <loc- in", label["suffix"])

def encode_actions(label, stats=None, encoding="xyzrotvec", i=None):
    """
    This code should be similar to that in gen_dataset.py
    """
    camera = DummyCamera(label["camera_intrinsic"], label["camera_extrinsic"])
    obj_start_pose = Pose(raw_pose=torch.tensor(label["obj_start_pose"]))
    obj_end_pose = Pose(raw_pose=torch.tensor(label["obj_end_pose"]))

    if encoding == "xyz":
        _, curve_3d = generate_curve_torch(obj_start_pose.get_p(), obj_end_pose.get_p(), num_points=2)
        curve_2d, depth, curve_tokens = encode_trajectory_xyz(curve_3d, camera)
    elif encoding == "xyzr":
        _, curve_3d = generate_curve_torch(obj_start_pose.get_p(), obj_end_pose.get_p(), num_points=2)
        curve_2d, depth, curve_tokens = encode_trajectory_xyz(curve_3d, camera)

    elif encoding == "xyzrotvec":
        from utils_traj_tokens import encode_trajectory_xyzrotvec as enc_func
        from utils_traj_tokens import decode_trajectory_xyzrotvec as dec_func
        from utils_trajectory import are_orns_close
        from gen_dataset import to_prefix_suffix
        grasp_pose = Pose(raw_pose=torch.tensor(label["grasp_pose"]))
        tcp_pose = Pose(raw_pose=torch.tensor(label["tcp_start_pose"]))
        action_text = label["action_text"]

        prefix, suffix, _, _, info = to_prefix_suffix(obj_start_pose, obj_end_pose, camera, grasp_pose, tcp_pose, action_text, enc_func)
        label["prefix"] = prefix
        label["suffix"] = suffix
        label["enc-info"] = info
        if i == 0:
            label["enc-info"]["mode"] = "xyzrotvec-cam-proj-abs"
        
        # check that the way we encoded is the same as what was done for the dataset creation
        check_encodings = False
        if check_encodings and label["suffix"] != curve_tokens:
            print("Warning: encodings differ")

        check_encode_decode = False
        if check_encode_decode:
            # by encoding and decoding, check the error that our encoding produces.
            curve_3d_est, orns_3d_est = decode_trajectory_xyzrotvec(curve_tokens, camera)
            if not torch.allclose(curve_3d, curve_3d_est, atol=.005):
                print("pos diff[m]", (curve_3d-curve_3d_est).abs().max())
            angles_close, max_angle = are_orns_close(orns_3d, orns_3d_est, return_max_diff=True)
            if not angles_close:
                print("angle diff[deg]", max_angle)
            
        # collect rotvec stats
        collect_rotvec_stats = True
        if collect_rotvec_stats:
            extrinsic_orn = R.from_matrix(camera.get_extrinsic_matrix()[:, :3, :3])
            extrinsic = Pose.create_from_pq(p=camera.get_extrinsic_matrix()[:, :3, 3],
                                            q=extrinsic_orn.as_quat(scalar_first=True))
            graps_pose_c = extrinsic * grasp_pose
            grap_pose_cR = R.from_quat(graps_pose_c.get_q(), scalar_first=True)
            rot_r = grap_pose_cR.as_rotvec()
            rot_e = grap_pose_cR.as_euler('xyz', degrees=True)
            stats["rots_r"].append(rot_r)
            stats["rots_e"].append(rot_e)
            stats["pos_s"].append(obj_start_pose.get_p().numpy())
            stats["pos_e"].append(obj_end_pose.get_p().numpy())
        curve_tokens = ""
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

filter_visible = True
LIMIT_LINES = 1250
#LIMIT_LINES = None
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
        lines_text = file.readlines()

    lines_text = lines_text[:LIMIT_LINES]

    # Assign to train-test
    random.seed(seed)
    indexes_all = np.arange(len(lines_text))
    random.shuffle(indexes_all)
    split_index = int(len(lines_text) * train_ratio)
    indices_train = indexes_all[:split_index]
    indices_valid = indexes_all[split_index:]

    stats = {"rots_r":[], "rots_e":[], "pos_s":[], "pos_e":[]}
    refered_files = []
    image2line = {}
    lines_new = ['', ]*len(lines_text)
    for i, line in tqdm(enumerate(lines_text), total=len(lines_text)):
        try:
            label = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f"Error decoding line {i}: {line[:50]}")
            continue

        #check_and_fix_format(label)
        encode_actions(label, stats, i=i)
        check_and_fix_format(label)

        if filter_visible:
            if label['enc-info']['didclip_traj']:# or label['enc-info']['didclip_tcp']:
                continue
            
        refered_files.append(label["image"])
        image2line[label["image"]] = i
        new_label = dict(image=label["image"], prefix=label["prefix"], suffix=label["suffix"])
        new_line = json.dumps(new_label) + "\n"
        lines_new[i] = new_line

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

    plot_stats = True
    if plot_stats:
        rots = np.array(stats["rots_r"])
        if rots.shape != (0,):
            print("r_0", np.percentile(rots[:, 0, 0], [.01, 99]).round(1))
            print("r_1", np.percentile(rots[:, 0, 1], [.01, 99]).round(1))
            print("r_2", np.percentile(rots[:, 0, 2], [.01, 99]).round(1))
        else:
            print("not stats collected")
        #import matplotlib.pyplot as plt
        #plt.hist(rots[:, 0, 2])
        #tmp = torch.cat(depths).flatten().tolist()
        #plt.hist(tmp)
        
    assert len(lines_new) == len(lines_text)
    lines_text = lines_new
    
    
    check_files = True
    if check_files:
        print("loading all .jpg images in", dataset_path)
        present_files = list(dataset_path.glob("*.jpg"))
        refered_files_set = set(refered_files)
        present_files_set = {file.name for file in present_files}
        missing_files = refered_files_set - present_files_set
        extra_files = present_files_set - refered_files_set
        print("referred but not present:", len(missing_files))
        print("present but not referred:", len(extra_files))
        assert len(missing_files) == 0

    def filter_save_subset(lines_text, subset_idxs, save_path, name_str):
        lines_train = lines_text[subset_idxs]
        len_before = len(lines_train)
        lines_train = lines_train[lines_train != '']
        len_after = len(lines_train)
        if len_before != len_after:
            print("filtered data", len_after/len_before)
        print(f"{name_str} images", len(lines_train))    
        with open(save_path, "w") as subset_file:
            subset_file.writelines(lines_train)
    
    lines_text = np.array(lines_text)
    filter_save_subset(lines_text, indices_train, train_path, "train")
    filter_save_subset(lines_text, indices_valid, valid_path, "valid")

    create_zip = False
    if create_zip:
        # create a zip of dataset_path.parent and place it in dataset_path.parent.parent
        print("\nCreating zip, this may take a while...")
        print(dataset_path.parent)
        source_dir = dataset_path.parent
        destination_dir = dataset_path.parent.parent / f"{source_dir.name}.zip"
        zip_path = shutil.make_archive(base_name=destination_dir.with_suffix(''), format='zip', root_dir=source_dir)
        print(f"Created zip at {zip_path} with size: {Path(zip_path).stat().st_size / (1024 * 1024):.2f} MB and {len(lines_text)} samples")

    copy_sample = True
    if copy_sample:
        for fn in refered_files[:10]:
            shutil.copy(dataset_path / fn , dataset_path.parent)

    set_trace()
    print("done.")

if __name__ == "__main__":
    dataset_path = Path("/tmp/clevr-act-6/dataset")
    split_dataset(dataset_path)