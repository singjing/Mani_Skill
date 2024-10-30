import gymnasium as gym
import numpy as np
import random
import sapien
import torch
import matplotlib.pyplot as plt

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers import RecordEpisode

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

import clevr_env  # to register env, not used otherwise
from utils_trajectory import project_points, generate_curve_torch, plot_gradient_curve
from utils_trajectory import subsample_trajectory
from utils_env_interventions import move_object_onto

from pdb import set_trace

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: int = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""


def reset_random(args, force=True):
    if args.seed is None or force:
        seed = random.randrange(2**32 - 1)
        args.seed = [seed]
    elif isinstance(args.seed, int):
        args.seed = [args.seed]
    np.random.seed(args.seed[0])


def clip_and_interpolate(curve_2d, camera):
    curve_2d_clip = curve_2d.clone()
    curve_2d_clip[:, :, 0] = torch.clip(curve_2d_clip[:, :, 0], 0+1, camera.width)
    curve_2d_clip[:, :, 1] = torch.clip(curve_2d_clip[:, :, 1], 0+1, camera.height)
    curve2d_short = subsample_trajectory(curve_2d_clip, points_new=5)
    return curve2d_short


def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    reset_random(args)
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        parallel_in_single_scene=parallel_in_single_scene,
        # **args.env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=env._max_episode_steps)

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    while True:
        obs, _ = env.reset(seed=args.seed[0], options=dict(reconfigure=True))
        if args.seed is not None:
            env.action_space.seed(args.seed[0])
        if args.render_mode is not None:
            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = args.pause
            env.render()
    
        env_id = 0
        # get before image
        from mani_skill.utils import common, sapien_utils

        #cam_pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        #env.base_env.scene.human_render_cameras['render_camera'].camera._render_cameras[0].set_local_pose(cam_pose)
        images = env.base_env.scene.get_human_render_camera_images('render_camera')
        image_before = images['render_camera'][env_id].numpy()

        # do intervention
        start_pose, end_pose, action_text = move_object_onto(env)

        # convert to trajectory
        camera = env.base_env.scene.human_render_cameras['render_camera'].camera
        _, curve_3d = generate_curve_torch(start_pose.get_p(), end_pose.get_p())
        curve_2d = project_points(camera, curve_3d)


        # get after image
        env.render()
        images = env.base_env.scene.get_human_render_camera_images('render_camera')
        image_after = images['render_camera'][env_id].numpy()

        # plot
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image_before)
        axs[1].imshow(image_after)
        #[x.axis('off') for x in axs]

        from utils_traj_tokens import traj2d_to_tokenstr
        curve_2d_short = clip_and_interpolate(curve_2d, camera)
        traj_token_str = traj2d_to_tokenstr(curve_2d_short[env_id], (camera.width, camera.height), "1")
        print("prefix", action_text)
        print("suffix", traj_token_str)

        x, y = curve_2d[env_id, :, 0].tolist(), curve_2d[env_id, :, 1].tolist()
        plot_gradient_curve(axs[0], x, y)
        axs[0].plot(curve_2d_short[env_id,:,0], curve_2d_short[env_id, :,1],'.-')

        fig.text(.05,.1, "^ "+action_text)
        plt.tight_layout()
        plt.show()
        
        # roll dice
        reset_random(args, force=True)
        print("done.")




    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info=env.step(action)
        if verbose:
            print("reward", reward)
            print("terminated", terminated)
            print("truncated", truncated)
            print("info", info)
        if args.render_mode is not None:
            env.render()
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
        


        def move_object(env, delta_pos=[0, 0, 0.1]):
            scene = env.base_env.scene
            object = scene.get_all_actors()[1]
            pose = object.pose
            pose.set_p(pose.p + delta_pos)
            object.set_pose(pose)
            #set_trace()

        #move_object(env)
        #move_camera(env)

    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
