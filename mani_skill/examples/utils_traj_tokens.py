# PaliGemma tokenize vocabulary with 1024 entries that represent coordinates in normalized image-space (<loc0000>...<loc1023>)
import numpy as np
import torch
from utils_trajectory import project_points, clip_and_interpolate
import re
from pdb import set_trace
from scipy.spatial.transform import Rotation as R

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


def parse_trajectory_xyz(caption, camera):
    num_tokens=3
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

def decode_trajectory_xyz(caption, camera):
    curve_25d = parse_trajectory_xyz(caption, camera)
    points_3d_est = unproject_points(camera, curve_25d) 
    return points_3d_est

def encode_trajectory_xyzr(curve_3d, orns_3d, camera, angle_in_world=True):
    """
    Trajectory encoded as y, x, z with x, y being pixel positions in range (0, 1023) and z being depth in cm
    """
    # In theory this code should handle (N, 7) poses, but for now we only support single envs
    DEPTH_SCALE = 100
    curve_25d = project_points(camera, curve_3d, return_depth=True)
    curve_2d = curve_25d[..., :2]
    depth = curve_25d[..., 2]
    curve_2d_short = clip_and_interpolate(curve_2d, camera)

    # transform orientation to matrix
    assert curve_3d.shape[:2] == orns_3d.shape[:2]
    assert curve_3d.ndim == 3 and orns_3d.ndim == 3
    assert curve_3d.shape[2] == 3 and orns_3d.shape[2] == 4

    if angle_in_world:
        euler_w = R.from_quat(orns_3d.view(-1, 4), scalar_first=True).as_euler('xyz', degrees=True)
        #assert np.allclose(euler_w[:, 0], 180, atol=1e-3)
        assert np.allclose(euler_w[:, 1], 0, atol=1e-3)
        assert np.all((euler_w[:, 2] >= 0) & (euler_w[:, 2] <= 180))    
        angle_rs = torch.tensor(euler_w[:, 2]).round().int()
    else:
        from mani_skill.utils.structs import Pose
        from utils_trajectory import batch_multiply
        poses = Pose.create_from_pq(p=curve_3d.view(-1, 3), q=orns_3d.view(-1, 4))
        set_trace()
        extrinsic_orn = R.from_matrix(camera.get_extrinsic_matrix()[:, :3, :3])
        extrinsic_p = Pose.create_from_pq(p=camera.get_extrinsic_matrix()[:, :3, 3],
                                          q=extrinsic_orn.as_quat(scalar_first=True))
        poses_c = extrinsic_p * poses
        poses_c_obj = R.from_quat(poses_c.get_q(), scalar_first=True)

        # double check points
        extrinsic_matrix = camera.get_extrinsic_matrix()  # Shape (3, 4)
        extrinsic_matrix = extrinsic_matrix.unsqueeze(0).expand(-1, curve_3d.shape[1], -1, -1) # shape (N, P, 3, 4)
        ones = torch.ones((*curve_3d.shape[:2], 1))  # Shape (N, P, 1)
        points_3d_h = torch.cat([curve_3d, ones], dim=-1)  # Shape (N, P, 4)
        points_camera = batch_multiply(extrinsic_matrix.view(-1, 3, 4), points_3d_h.view(-1, 4))
        is_close = np.allclose(poses_c.get_p(),points_camera, atol=1e-3)
        print(poses_c_obj.as_euler('xyz', degrees=True))
        set_trace()

    # This is the part that is not parrelized
    env_idx = 0
    depth_env = (depth[env_idx]*DEPTH_SCALE).round().int()  # distance from camera in [cm]
    traj_1024 = normalize_imgcoord(curve_2d_short[env_idx], (camera.width, camera.height))
    result = ""
    for keypoint_xy, keypoint_d, roll in zip(traj_1024, depth_env, angle_rs):
        result += f"<loc{keypoint_xy[1]:04d}><loc{keypoint_xy[0]:04d}><loc{keypoint_d:04d}><loc{roll:04d}>"
    return curve_2d_short, depth, result.strip()  # Remove any trailing newlines

def parse_trajectory_xyzr(caption, camera):
    num_tokens=4
    DEPTH_SCALE = 100
    # Pattern to extract numbers inside <loc####> tags
    loc_strings = re.findall(r"<loc(\d{4})>", caption)
    num_position_tokens = len(loc_strings)
    loc_strings_pairs = loc_strings[:(num_position_tokens//num_tokens)*num_tokens]
    loc_numbers = [int(x) for x in loc_strings_pairs]
    loc_h = [x/(1024-1)*camera.height for x in loc_numbers[::num_tokens]]
    loc_w = [x/(1024-1)*camera.width for x in loc_numbers[1::num_tokens]]
    loc_d = [x/DEPTH_SCALE for x in loc_numbers[2::num_tokens]]  # depth
    loc_r = [x for x in loc_numbers[3::num_tokens]]  # roll
    curve_25d = torch.tensor((loc_w, loc_h, loc_d)).T
    euler_w = torch.tensor([[180, 0, x] for x in loc_r])
    quat_w = torch.tensor(R.from_euler('xyz', euler_w, degrees=True).as_quat(scalar_first=True)).float()
    return curve_25d, quat_w  # shape (P, 3 = u, v, d)

def decode_trajectory_xyzr(caption, camera):
    curve_25d, orns_3d_est = parse_trajectory_xyzr(caption, camera)
    points_3d_est = unproject_points(camera, curve_25d)
    return points_3d_est, orns_3d_est


def encode_trajectory_xyzrotvec(curve_3d, orns_3d, camera):
    """
    Trajectory encoded as y, x, z with x, y being pixel positions in range (0, 1023) and z being depth in cm
    """
    # In theory this code should handle (N, 7) poses, but for now we only support single envs
    DEPTH_SCALE = 100
    ROT_SCALE = 100
    max_rotvec = np.pi/2

    curve_25d = project_points(camera, curve_3d, return_depth=True)
    curve_2d = curve_25d[..., :2]
    depth = curve_25d[..., 2]
    curve_2d_short = clip_and_interpolate(curve_2d, camera)

    # transform orientation to matrix
    assert curve_3d.shape[:2] == orns_3d.shape[:2]
    assert curve_3d.ndim == 3 and orns_3d.ndim == 3
    assert curve_3d.shape[2] == 3 and orns_3d.shape[2] == 4

    from mani_skill.utils.structs import Pose
    poses = Pose.create_from_pq(p=curve_3d.view(-1, 3), q=orns_3d.view(-1, 4))
    extrinsic_orn = R.from_matrix(camera.get_extrinsic_matrix()[:, :3, :3])
    extrinsic_p = Pose.create_from_pq(p=camera.get_extrinsic_matrix()[:, :3, 3],
                                        q=extrinsic_orn.as_quat(scalar_first=True))
    poses_c = extrinsic_p * poses
    orn_c_obj = R.from_quat(poses_c.get_q(), scalar_first=True)

    rotvec = torch.tensor(orn_c_obj.as_rotvec())
    rotvec_positive = rotvec * torch.tensor([-1, 1, 1])
    assert (rotvec_positive > 0).all() and (rotvec_positive < max_rotvec).all()

    # This is the part that is not parrelized
    env_idx = 0
    depth_env = (depth[env_idx]*DEPTH_SCALE).round().int()  # distance from camera in [cm]
    traj_1024 = normalize_imgcoord(curve_2d_short[env_idx], (camera.width, camera.height))
    rotvec_int = (rotvec_positive * ROT_SCALE).round().int()
    result = ""
    for keypoint_xy, keypoint_d, rv in zip(traj_1024, depth_env, rotvec_int):
        result += f"<loc{keypoint_xy[1]:04d}><loc{keypoint_xy[0]:04d}><loc{keypoint_d:04d}>"
        result += f"<loc{rv[0]:04d}><loc{rv[1]:04d}><loc{rv[2]:04d}>"
    return curve_2d_short, depth, result.strip()  # Remove any trailing newlines

from utils_trajectory import unproject_points
from scipy.spatial.transform import Rotation as R
from mani_skill.utils.structs import Pose

def decode_trajectory_xyzrotvec(caption, camera):
    num_tokens = 6
    DEPTH_SCALE = 100
    ROT_SCALE = 100

    # Pattern to extract numbers inside <loc####> tags
    loc_strings = re.findall(r"<loc(\d{4})>", caption)
    num_position_tokens = len(loc_strings)
    loc_strings_pairs = loc_strings[:(num_position_tokens//num_tokens)*num_tokens]
    loc_numbers = [int(x) for x in loc_strings_pairs]
    loc_h = [x/(1024-1)*camera.height for x in loc_numbers[::num_tokens]]
    loc_w = [x/(1024-1)*camera.width for x in loc_numbers[1::num_tokens]]
    loc_d = [x/DEPTH_SCALE for x in loc_numbers[2::num_tokens]]  # depth
    loc_r0 = [x/ROT_SCALE for x in loc_numbers[3::num_tokens]]  # rotvec[0]
    loc_r1 = [x/ROT_SCALE for x in loc_numbers[4::num_tokens]]  # rotvec[1]
    loc_r2 = [x/ROT_SCALE for x in loc_numbers[5::num_tokens]]  # rotvec[2]
    
    curve_25d = torch.tensor((loc_w, loc_h, loc_d)).T
    rotvec_positive = torch.tensor((loc_r0, loc_r1, loc_r2)).T
    rotvec = rotvec_positive * torch.tensor([-1, 1, 1])
    quat_c = torch.tensor(R.from_rotvec(rotvec).as_quat(scalar_first=True)).float()

    # from camera to world coordinates
    extrinsic_orn = R.from_matrix(camera.get_extrinsic_matrix()[:, :3, :3])
    extrinsic = Pose.create_from_pq(p=camera.get_extrinsic_matrix()[:, :3, 3],
                                    q=extrinsic_orn.as_quat(scalar_first=True))
    quat_w = extrinsic.inv() * Pose.create_from_pq(q=quat_c)
    curve_w = unproject_points(camera, curve_25d) 

    return curve_w, quat_w.get_q().unsqueeze(0)  # shape (P, 3 = u, v, d)


def are_orns_close(orns_3d, orns_3d_est, tol_degrees=0.5):
    orns_R = R.from_quat(orns_3d.view(-1,4), scalar_first=True)
    orns_est_R = R.from_quat(orns_3d_est.view(-1,4), scalar_first=True)
    magnitude_radians = torch.tensor((orns_est_R * orns_R.inv()).magnitude())
    angle_degrees = magnitude_radians * (180.0 / torch.pi)
    return torch.allclose(angle_degrees, torch.zeros_like(angle_degrees), atol=tol_degrees)


def check_encode_decode():
    """
    Check that encoding a trajectory to tokens and back does not produce large errors.
    """
    from utils_trajectory import DummyCamera
    from utils_trajectory import unproject_points
    
    camera_extrinsic = [[[-0.759, 0.651, 0.0, 0.0], [0.301, 0.351, -0.887, 0.106], [-0.577, -0.673, -0.462, 0.575]]]
    camera_intrinsic = [[[410.029, 0.0, 224.0], [0.0, 410.029, 224.0], [0.0, 0.0, 1.0]]]

    camera = DummyCamera(camera_intrinsic, camera_extrinsic)
    points_3d = torch.tensor([[[-0.1689,  0.0338,  0.0350],
                               [-0.1137,  0.1394,  0.0700]]])
    
    orns_3d = torch.tensor([[[0.0000, 0.4484, 0.8939, 0.0000],
                             [0.0000, 0.4484, 0.8939, 0.0000]]])

    # check xyz
    curve_25d, depth, token_str = encode_trajectory_xyz(points_3d, camera)
    points_3d_est = decode_trajectory_xyz(token_str, camera)
    assert torch.allclose(points_3d, points_3d_est, atol=.005)
    
    # check xyzr
    curve_25d, depth, token_str = encode_trajectory_xyzr(points_3d, orns_3d, camera)
    points_3d_est, orns_3d_est = decode_trajectory_xyzr(token_str, camera)
    assert torch.allclose(points_3d, points_3d_est, atol=.005)
    assert torch.allclose(orns_3d, orns_3d_est, atol=.005)
    
    # check xyzr
    curve_25d, depth, token_str = encode_trajectory_xyzrotvec(points_3d, orns_3d, camera)
    points_3d_est, orns_3d_est = decode_trajectory_xyzrotvec(token_str, camera)
    assert torch.allclose(points_3d, points_3d_est, atol=.005)
    assert torch.allclose(orns_3d, orns_3d_est, atol=.005)
    

if __name__ == "__main__":
    check_encode_decode()
    print("All tests passed!")