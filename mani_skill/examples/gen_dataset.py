"""
Option 1: record inital frames:
python gen_dataset.py -e "ClevrMove-v1"   --render-mode="rgb_array" 

Option 2: record trajectories
python gen_dataset.py -e "ClevrMove-v1"   --render-mode="rgb_array" -c "pd_joint_pos"
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

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers import RecordEpisode

import clevr_env  # to register env, not used otherwise
from utils_env_interventions import move_object_onto
from utils_trajectory import project_points, generate_curve_torch, plot_gradient_curve
from utils_trajectory import clip_and_interpolate
from utils_traj_tokens import encode_trajectory_xy, encode_trajectory_xyz

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


def save_solve_policy_to_json(env, args, camera, json_dict, vis):
    #from mani_skill.examples.motionplanning.panda.solutions import solveStackCube
    #solve = solveStackCube
    from clevr_env_solver import solve
    #try:
    res = solve(env, seed=args.seed[0], debug=False, vis=vis, dry_run=True)
    #except Exception as e:
    #    print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
    #    res = -1

    traj_poses_3d = []
    traj_orns_3d = []
    traj_gripper = []
    for wp in res:
        if isinstance(wp, sapien.Pose):
            traj_poses_3d.append(wp.get_p())
            traj_orns_3d.append(wp.get_q())
        elif wp == "close_gripper" or wp == "open_gripper":
            gripper_act = 0 if wp == "close_gripper" else 1
            gripper_act_n = 1 if wp == "close_gripper" else 0
            if traj_gripper == []:
                traj_gripper.extend([gripper_act_n]*(len(traj_poses_3d)-1))
                traj_gripper.append(gripper_act)
            else:
                traj_gripper.extend([traj_gripper[-1]]*(len(traj_poses_3d)-len(traj_gripper)-1))
                traj_gripper.append(gripper_act)

    traj_poses_3d = np.array(traj_poses_3d).tolist()
    traj_orns_3d = np.array(traj_orns_3d).tolist()
    traj_gripper = np.array(traj_gripper).tolist()
    scene_data = dict(traj_p=traj_poses_3d, traj_q=traj_orns_3d, traj_g=traj_gripper)
    json_dict.update(scene_data)          

    # plot the trajectory
    # curve_2d = project_points(camera, torch.tensor([traj_poses_3d]))
    # x, y = curve_2d[env_idx, :, 0].tolist(), curve_2d[env_idx, :, 1].tolist()
    # plot_gradient_curve(axs[0], x, y)

def main(args: Args, plot=True):
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

    vis = False
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
        start_pose, end_pose, action_text = move_object_onto(env, pretend=True)
        env.unwrapped.set_goal_pose(end_pose)
        
        # convert to default trajectory
        camera = env.base_env.scene.human_render_cameras['render_camera'].camera
        _, curve_3d = generate_curve_torch(start_pose.get_p(), end_pose.get_p())
        traj_token_str = encode_trajectory_xy(curve_3d, camera)

        json_dict = dict(prefix=action_text, suffix=traj_token_str,
                         camera_extrinsic=camera.get_extrinsic_matrix().detach().numpy().tolist(),
                         camera_intrinsic=camera.get_intrinsic_matrix().detach().numpy().tolist(),
                         start_pose=start_pose.raw_pose.detach().numpy().tolist(),
                         end_pose=end_pose.raw_pose.detach().numpy().tolist())
        save_solve_policy_to_json(env, args, camera, json_dict, vis)

        if action_text.split()[3] == "sphere":
            # TODO(max): Find a good angle to do grasp
            print(env.unwrapped.agent.tcp.pose.get_q()[0])
            print(torch.tensor(json_dict['traj_q'][0]))

            from scipy.spatial.transform import Rotation as R
            from mani_skill.utils.structs.pose import Pose

            rot = R.from_matrix(env.unwrapped.agent.tcp.pose.to_transformation_matrix()[0, :3, :3].numpy())
            print(rot.as_euler('xyz', degrees=True))
            
            rot2 = R.from_matrix(Pose.create_from_pq(p=None, q=torch.tensor(json_dict['traj_q'][0])).to_transformation_matrix()[0, :3, :3].numpy())
            print(rot2.as_euler('xyz', degrees=True))

            #set_trace()

        _, curve_3d = generate_curve_torch(start_pose.get_p(), end_pose.get_p(), num_points=2)
        orns_3d = torch.tensor(json_dict['traj_q'][0])
        orns_3d = orns_3d.expand(curve_3d.shape[0], curve_3d.shape[1], -1)
        
        encode_decode_trajectory = False
        if encode_decode_trajectory:
            # Option 2: encode trajectory as xyz
            from utils_traj_tokens import parse_trajectory_xyz
            from utils_trajectory import unproject_points
            curve_25d, depth, token_str = encode_trajectory_xyz(curve_3d, camera)
            curve_25d = parse_trajectory_xyz(token_str, camera, num_tokens=3)
            curve_3d_est = unproject_points(camera, curve_25d) 
            curve_3d = curve_3d_est # set the unparsed trajectory one used for policy

        # Evaluate the trajectory
        from mani_skill.utils.structs import Pose
        from mani_skill.examples.motionplanning.panda.motionplanner import \
            PandaArmMotionPlanningSolver
        
        #set_trace()
        _, curve_3d_i = generate_curve_torch(start_pose.get_p(), end_pose.get_p(), num_points=3)
        #orn = env.unwrapped.agent.tcp.pose.get_q()
        grasp_pose = Pose.create_from_pq(p=curve_3d[:, 0], q=orns_3d[:, 0])
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
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
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        planner.move_to_pose_with_screw(lift_pose)
        planner.move_to_pose_with_screw(align_pose)
        res = planner.open_gripper()
        print("reward", env.unwrapped.eval_reward())
        planner.close()

        #curve_3d_parse = decode_trajectory_xyz(traj_xyz_str, camera)
        #print("prefix", action_text)
        #print("suffix", action_text)
        #set_trace()

        plot = False
        if plot:
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

        if args.record_dir:
            env.flush_trajectory(save=True)
            env.flush_video(name=new_traj_name, save=True)

        # roll dice
        reset_random(args, force=True)
        #yield image_before, json_dict, args.seed[0]

    env.close()


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

    annotations = []
    for i in tqdm(range(N)):
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

    #Important: in order to save dataset uncomment the yeild line in the main function.
    # python gen_dataset.py -e "ClevrMove-v1" --render-mode="rgb_array"
    #N_samples = 100000/0.8
    #save_dataset(main(parsed_args, plot=False), N=int(N_samples), dataset_path=Path("/tmp/clevr-act-2/dataset"))