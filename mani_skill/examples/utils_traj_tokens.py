# PaliGemma tokenize vocabulary with 1024 entries that represent coordinates in normalized image-space (<loc0000>...<loc1023>)
import numpy as np
import torch
from utils_trajectory import project_points, clip_and_interpolate
import re
from pdb import set_trace

def normalize_imgcoord(traj, resolution_wh):
    """
    Normalize (xy) pixel positions into (0, 1023) following paligemma detection
    Arguments:
        traj: array with shape (P waypoints, 2) with waypoints in (width, height)
    Returns:
        traj_1024: array with shape (P waypoints, 2) with waypoints normalized
    """
    assert traj.shape[1] == 2
    w, h = resolution_wh
    # Convert from resolution_wh back to 1024x1024 space
    traj_1024 = traj / np.array([[w, h]]) * (1024 - 1)
    traj_1024 = traj_1024.round().int()
    if torch.any(traj_1024 != torch.clamp(traj_1024, 0, 1024 - 1)):
        print("Warning: tokens out of range.")
        # Construct the result string using the original pattern
    return traj_1024


def encode_trajectory_xy(curve_3d, camera, num_points=5, end_string="1"):
    """convert bbox (xyxy) format into paligemma token string
    Arguments:
        traj: array with shape (N waypoints, 2)"""
    curve_25d = project_points(camera, curve_3d)
    curve_2d = curve_25d[..., :2]
    curve_2d_short = clip_and_interpolate(curve_2d, camera, num_points=num_points)
    assert len(curve_2d_short) == 1  # Only support single envs for now 
    env_idx = 0
    traj_1024 = normalize_imgcoord(curve_2d_short[env_idx], (camera.width, camera.height))
    result = ""
    for keypoint in traj_1024:
        result += f"<loc{keypoint[1]:04d}><loc{keypoint[0]:04d}>"
    result +=  f" {end_string}"
    return result.strip()  # Remove any trailing newlines


def decode_trajectory_xy(caption, camera):
    # Pattern to extract numbers inside <loc####> tags
    loc_strings = re.findall(r"<loc(\d{4})>", caption)
    num_position_tokens = len(loc_strings)
    loc_strings_pairs = loc_strings[:(num_position_tokens//2)*2]
    loc_numbers = [int(x) for x in loc_strings_pairs]
    loc_h = [x/(1024-1)*camera.height for x in loc_numbers[::2]]
    loc_w = [x/(1024-1)*camera.width for x in loc_numbers[1::2]]
    curve_2d = np.stack((loc_w, loc_h), axis=1)
    return curve_2d


def encode_trajectory_xyz(curve_3d, camera):
    """
    Trajectory encoded as y, x, z with x, y being pixel positions in range (0, 1023) and z being depth in cm
    """
    # In theory this code should handle (N, 7) poses, but for now we only support single envs
    DEPTH_SCALE = 100
    curve_25d = project_points(camera, curve_3d, return_depth=True)
    curve_2d = curve_25d[..., :2]
    depth = curve_25d[..., 2]
    curve_2d_short = clip_and_interpolate(curve_2d, camera)
    # This is the part that is not parrelized
    env_idx = 0
    depth_env = (depth[env_idx]*DEPTH_SCALE).round().int()  # distance from camera in [cm]
    traj_1024 = normalize_imgcoord(curve_2d_short[env_idx], (camera.width, camera.height))
    result = ""
    for keypoint_xy, keypoint_d in zip(traj_1024, depth_env):
        result += f"<loc{keypoint_xy[1]:04d}><loc{keypoint_xy[0]:04d}><loc{keypoint_d:04d}>"
    return curve_2d_short, depth, result.strip()  # Remove any trailing newlines


def parse_trajectory_xyz(caption, camera, num_tokens=3):
    DEPTH_SCALE = 100
    # Pattern to extract numbers inside <loc####> tags
    loc_strings = re.findall(r"<loc(\d{4})>", caption)
    num_position_tokens = len(loc_strings)
    loc_strings_pairs = loc_strings[:(num_position_tokens//num_tokens)*num_tokens]
    loc_numbers = [int(x) for x in loc_strings_pairs]
    loc_h = [x/(1024-1)*camera.height for x in loc_numbers[::num_tokens]]
    loc_w = [x/(1024-1)*camera.width for x in loc_numbers[1::num_tokens]]
    loc_d = [x/DEPTH_SCALE for x in loc_numbers[2::num_tokens]]  # depth
    curve_25d = torch.tensor((loc_w, loc_h, loc_d)).T
    return curve_25d  # shape (P, 3 = u, v, d)

from utils_trajectory import unproject_points

def decode_trajectory_xyz(caption, camera):
    curve_25d = parse_trajectory_xyz(caption, camera, num_tokens=3)
    curve_3d = unproject_points(camera, curve_25d) 

    return curve_3d


def check_encode_decode():
    """
    Check that encoding a trajectory to tokens and back does not produce large errors.
    """
    from utils_trajectory import DummyCamera
    camera_extrinsic = [[[-0.759, 0.651, 0.0, 0.0], [0.301, 0.351, -0.887, 0.106], [-0.577, -0.673, -0.462, 0.575]]]
    camera_intrinsic = [[[410.029, 0.0, 224.0], [0.0, 410.029, 224.0], [0.0, 0.0, 1.0]]]

    camera = DummyCamera(camera_intrinsic, camera_extrinsic)
    points_3d = torch.tensor([[[-0.1689,  0.0338,  0.0350],
                               [-0.1137,  0.1394,  0.0700]]])
    
    curve_25d, depth, token_str = encode_trajectory_xyz(points_3d, camera)
    curve_25d = parse_trajectory_xyz(token_str, camera, num_tokens=3)
    points_3d_est = unproject_points(camera, curve_25d) 
    is_close = torch.allclose(points_3d, points_3d_est, atol=.005)
    assert is_close
    
    
if __name__ == "__main__":
    check_encode_decode()
    print("All tests passed!")