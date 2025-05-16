import numpy as np
import subprocess
from typing import Tuple, Dict


def get_git_commit_hash() -> str:
    """Returns the current git commit hash as a string."""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return commit_hash
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to get git commit hash.")


def check_no_uncommitted_changes():
    """Raises an error if there are uncommitted changes in the repo."""
    try:
        status = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        if status:
            raise RuntimeError("Uncommitted changes detected. Please commit or stash before proceeding.")
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to check git status.")


def downcast_seg_array(env):
    try:
        seg = env._trajectory_buffer.observation['sensor_data']['render_camera']['segmentation']
        if seg.max() <= 255:
            seg = seg.astype(np.uint8)
        env._trajectory_buffer.observation['sensor_data']['render_camera']['segmentation'] = seg
    except KeyError:
        pass


def check_object_pixels(seg_image, obs_scene, N_percent, return_percent=False) -> Tuple[bool, Dict]:
    """
    Checks if each relevant object in obs_scene has at least N% of the image pixels.
    Uses np.unique for efficiency.

    Parameters:
    - seg_image (np.ndarray): 2D array of segmentation labels.
    - obs_scene (dict): Scene info with 'object_info' a dict with seg_id, task_req
    - N_percent (float): Minimum percentage (0–100) of total pixels required per relevant object.

    Returns:
    - bool: True if all relevant objects have at least N% of total pixels, False otherwise.
    """
    assert return_percent
    object_info = obs_scene.get('object_info', {})
    total_pixels = seg_image.size

    # Filter relevant object IDs (ignore ground/table and relevance != 1)
    relevant_ids = {
        cur_info['seg_id'] for obj_name, cur_info in object_info.items()
        if cur_info['task_req'] == 1 and obj_name not in ['ground', 'table-workspace']
    }

    # Count pixels using np.unique
    unique_ids, counts = np.unique(seg_image, return_counts=True)
    id_to_percent = dict(zip(unique_ids, counts / total_pixels * 100))

    vis_percent = [id_to_percent.get(obj_id, 0) for obj_id in relevant_ids]
    all_true = bool(np.all([x > N_percent for x in vis_percent]))
    return all_true, id_to_percent


def apply_check_object_pixels(env, N_percent):
    """
    Applies check_object_pixels to each observation in the environment's trajectory buffer.
    """
    n_env = env._trajectory_buffer.observation['sensor_data']['render_camera']['segmentation'].shape[0]
    assert n_env == 1

    env_id = 0
    frame_id = 0  # we check the first frame for visibility of objects
    seg_image = env._trajectory_buffer.observation['sensor_data']['render_camera']['segmentation'][env_id, frame_id]
    assert seg_image.ndim == 3
    assert seg_image.shape[2] == 1  # one channel

    obs_scene = env.unwrapped.get_obs_scene()
    are_visible, id_to_percent = check_object_pixels(seg_image, obs_scene, N_percent, return_percent=True)
    # move data around, so that we can all get_obs_scene in Recorder
    env.unwrapped.seg_id_to_initial_frame_percent = id_to_percent
    return are_visible


def apply_check_object_pixels_obs(observation, env, N_percent):
    """
    Applies check_object_pixels to each observation in the environment's trajectory buffer.
    """
    n_env = observation['sensor_data']['render_camera']['segmentation'].shape[0]
    assert n_env == 1
    seg_image = observation['sensor_data']['render_camera']['segmentation'][0].detach().cpu().numpy()
    assert seg_image.ndim == 3
    assert seg_image.shape[2] == 1  # one channel
    obs_scene = env.unwrapped.get_obs_scene()
    are_visible, id_to_percent = check_object_pixels(seg_image, obs_scene, N_percent, return_percent=True)
    env.unwrapped.seg_id_to_initial_frame_percent = id_to_percent
    return are_visible
