"""
Option 1: record inital frames:
python gen_dataset.py -e "ClevrMove-v1" --render-mode="rgb_array" 

Option 2: record trajectories
python gen_dataset.py -e "ClevrMove-v1" --render-mode="rgb_array" -c "pd_joint_pos"
"""
import gymnasium as gym
import numpy as np
import random
import sapien
import torch
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.structs import Pose
        
import clevr_env  # do import to register env, not used otherwise
from clevr_env_solver import get_grasp_pose_and_obb
from utils_env_interventions import move_object_onto
from utils_trajectory import project_points, generate_curve_torch, plot_gradient_curve
from utils_trajectory import clip_and_interpolate

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



def main(args: Args, vis=True):
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
        sample_name = str(args.seed[0]).zfill(10)
        new_traj_name = f"CLEVR_{sample_name}"
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, trajectory_name=new_traj_name, info_on_video=False, save_trajectory=True, max_steps_per_video=env._max_episode_steps)

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)
    print("Render mode", args.render_mode)

    for _ in range(10**6):
        obs, _ = env.reset(seed=args.seed[0], options=dict(reconfigure=True))
        if args.seed is not None:
            env.action_space.seed(args.seed[0])
        if vis and args.render_mode is not None:
            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = args.pause
            env.render()
        else:
            env.render()
        env_idx = 0

        # in theory randomize camera position
        #cam_pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        #env.base_env.scene.human_render_cameras['render_camera'].camera._render_cameras[0].set_local_pose(cam_pose)
        
        # get before image
        images = env.base_env.scene.get_human_render_camera_images('render_camera')
        image_before = images['render_camera'][env_idx].numpy()

        # do intervention ( this will set env.base_env.cubeA / cubeB)
        obj_start, obj_end, action_text = move_object_onto(env, pretend=True)
        env.unwrapped.set_goal_pose(obj_end)
        from mani_skill.utils.structs import Pose
        grasp_pose_sapien, _ = get_grasp_pose_and_obb(env)
        # sapien to mani_skill
        grasp_pose = Pose.create_from_pq(p=grasp_pose_sapien.get_p(), q=grasp_pose_sapien.get_q())

        # convert to default trajectory
        camera = env.base_env.scene.human_render_cameras['render_camera'].camera
        
        # this is the orignal encoding
        #_, curve_3d = generate_curve_torch(obj_start.get_p(), obj_end.get_p())
        #token_str = encode_trajectory_xy(curve_3d, camera)

        _, curve_3d = generate_curve_torch(obj_start.get_p(), obj_end.get_p(), num_points=2)
        orns_3d = grasp_pose.get_q().clone().detach()  # get rotation
        orns_3d = orns_3d.expand(curve_3d.shape[0], curve_3d.shape[1], -1)
        from utils_traj_tokens import encode_trajectory_xyzrotvec, decode_trajectory_xyzrotvec
        curve_25d, depth, token_str = encode_trajectory_xyzrotvec(curve_3d, orns_3d, camera)

        # encode tcp position
        tcp_pose = env.unwrapped.agent.tcp.pose
        _, _, tcp_str = encode_trajectory_xyzrotvec(tcp_pose.get_p().unsqueeze(0), tcp_pose.get_q().unsqueeze(0), camera)   
        
        json_dict = dict(prefix=action_text+" "+tcp_str, suffix=token_str,
                         action_text=action_text,
                         camera_extrinsic=camera.get_extrinsic_matrix().detach().numpy().tolist(),
                         camera_intrinsic=camera.get_intrinsic_matrix().detach().numpy().tolist(),
                         obj_start_pose=obj_start.raw_pose.detach().numpy().tolist(),
                         obj_end_pose=obj_end.raw_pose.detach().numpy().tolist(),
                         tcp_start_pose=env.unwrapped.agent.tcp.pose.raw_pose.detach().numpy().tolist(),
                         grasp_pose=grasp_pose.raw_pose.detach().numpy().tolist())
        
        encode_decode_trajectory = False
        if encode_decode_trajectory:
            from utils_traj_tokens import encode_trajectory_xyzrotvec, decode_trajectory_xyzrotvec
            #curve_25d, depth, token_str = encode_trajectory_xyzrotvec(curve_3d, orns_3d, camera)
            curve_3d_est, orns_3d_est = decode_trajectory_xyzrotvec(token_str, camera)
            curve_3d = curve_3d_est  # set the unparsed trajectory one used for policy
            orns_3d = orns_3d_est

        # Evaluate the trajectory
        eval_trajectory = False
        if eval_trajectory:
            from mani_skill.examples.motionplanning.panda.motionplanner import \
                PandaArmMotionPlanningSolver
            
            assert curve_3d.shape[1] == 2 and orns_3d.shape[1] == 2  # start and stop poses
            _, curve_3d_i = generate_curve_torch(curve_3d[:, 0], curve_3d[:, -1], num_points=3)
            grasp_pose_sapien = Pose.create_from_pq(p=curve_3d[:, 0], q=orns_3d[:, 0])
            reach_pose = grasp_pose_sapien * sapien.Pose([0, 0, -0.05])
            lift_pose = Pose.create_from_pq(p=curve_3d_i[:, 1], q=orns_3d[:, 1])
            align_pose = Pose.create_from_pq(p=curve_3d_i[:, 2], q=orns_3d[:, 1])

            planner = PandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=vis,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=vis,
                print_env_info=False,
            )
            planner.move_to_pose_with_screw(reach_pose)
            planner.move_to_pose_with_screw(grasp_pose_sapien)
            planner.close_gripper()
            planner.move_to_pose_with_screw(lift_pose)
            planner.move_to_pose_with_screw(align_pose)
            res = planner.open_gripper()
            print("reward", env.unwrapped.eval_reward())
            planner.close()

        #test_angles(env, camera)
        #plot_curve(env, curve_3d, camera, env_idx, action_text, image_before)

        if args.record_dir:
            env.flush_trajectory(save=True)
            env.flush_video(name=new_traj_name, save=True)

        # roll dice
        reset_random(args, force=True)
        #yield image_before, json_dict, args.seed[0]

    env.close()


def plot_curve(env, curve_3d, camera, env_idx, action_text, image_before):
    curve_2d = project_points(camera, curve_3d)
    curve_2d_short = clip_and_interpolate(curve_2d, camera)

    # get after image
    env.render()
    images = env.base_env.scene.get_human_render_camera_images('render_camera')
    image_after = images['render_camera'][env_idx].numpy()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image_before)
    axs[1].imshow(image_after)
    #[x.axis('off') for x in axs]

    x, y = curve_2d[env_idx, :, 0].tolist(), curve_2d[env_idx, :, 1].tolist()
    plot_gradient_curve(axs[0], x, y)
    axs[0].plot(curve_2d_short[env_idx,:,0], curve_2d_short[env_idx, :,1],'.-')

    fig.text(.05,.1, "^ "+action_text)
    plt.tight_layout()
    plt.show()


def test_angles(env, camera):
    from mani_skill.utils.structs import Pose
    from mani_skill.examples.motionplanning.panda.motionplanner import \
        PandaArmMotionPlanningSolver
    from mani_skill.utils.structs.pose import to_sapien_pose
    from mani_skill.utils.structs.pose import to_sapien_pose
    import time
    tcp = env.unwrapped.agent.tcp.pose

    vis = True
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    extrinsic_orn = R.from_matrix(camera.get_extrinsic_matrix()[:, :3, :3])
    extrinsic = Pose.create_from_pq(p=camera.get_extrinsic_matrix()[:, :3, 3],
                                    q=extrinsic_orn.as_quat(scalar_first=True))
    # rotate gripper in camera frame
    for x in range(-60, 60, 1):
        angle = R.from_euler('xyz', [0, 0, x], degrees=True)
        pose_c = Pose.create_from_pq(p=[0,0,.3], q=angle.as_quat(scalar_first=True))
        # pose_c = extrinsic * pose_w
        pose_w = extrinsic.inv() * pose_c
        pose_ws = to_sapien_pose(pose_w)
        planner.grasp_pose_visual.set_pose(pose_ws)
        planner.render_wait()
        env.base_env.render_human()
        print('xxx', angle)
        #planner.move_to_pose_with_screw(pose)
        time.sleep(.3)

    # rotate gripper in world frame
    for angle_z in range(-60, 60, 10):
        angle = R.from_euler('xyz', [180+angle_z, 0, 0], degrees=True)
        pose = Pose.create_from_pq(p=tcp.get_p(), q=angle.as_quat(scalar_first=True))
        planner.move_to_pose_with_screw(pose)
        time.sleep(.3)

    for angle_y in range(-60, 60, 10):
        angle = R.from_euler('xyz', [180, angle_y, 0], degrees=True)
        pose = Pose.create_from_pq(p=tcp.get_p(), q=angle.as_quat(scalar_first=True))
        planner.move_to_pose_with_screw(pose)
        time.sleep(.3)


def save_dataset(sample_generator, N: int, dataset_path):
    os.makedirs(dataset_path, exist_ok=True)
    # Initialize a ThreadPoolExecutor with one or more workers
    executor = ThreadPoolExecutor(max_workers=3)
    def save_image(image, path) -> None:
        Image.fromarray(image).save(path, format='JPEG')
        
    def save_image_async(image, path) -> None:
        # Submit the save_image task to be run in a separate thread
        executor.submit(save_image, image, path)

    def save_labels(annotations):
        json_file = dataset_path / "_annotations.all.jsonl"
        # "a" means append to the file
        with open(json_file, "a") as f:
            for obj in annotations:
                json.dump(obj, f)
                f.write("\n")

    def get_num_lines():
        json_file = dataset_path / "_annotations.all.jsonl"
        if Path(json_file).exists():
            with open(json_file) as f:
                return sum(1 for line in f)
        else:
            return 0
        
    N_cur = get_num_lines()
    N_remaining = N - N_cur
    annotations = []
    for i in tqdm(range(N_remaining)):
        image_before, json_dict, rnd_seed = next(sample_generator)
        sample_name = str(rnd_seed).zfill(10)
        image_filename = f"CLEVR_{sample_name}.jpg"
        json_dict["image"] = image_filename
        save_image_async(image_before, dataset_path / image_filename)
        annotations.append(json_dict)
        if i % 100 == 0:
            save_labels(annotations)
            annotations = []
    save_labels(annotations)
    print("done.")
        

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    #Important: in order to save dataset comment the yeild line in the main function.
    main(parsed_args)

    # python gen_dataset.py -e "ClevrMove-v1" --render-mode="rgb_array"
    #Important: in order to save dataset do the following:
    # 1. uncomment the yeild line in the main function.
    # 2. set eval_trajectory = False
    # 3. set encode_decode_trajectory = False
    #N_samples = 200000/0.8
    #save_dataset(main(parsed_args, vis=False), N=int(N_samples), dataset_path=Path("/tmp/clevr-act-6/dataset"))