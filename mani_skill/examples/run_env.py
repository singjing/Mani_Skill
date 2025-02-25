"""
Option 1: record inital frames:
python run_env.py 

Option 2: record trajectories (e.g. with rt shader)
python run_env.py --record_dir /tmp/clevr-obs-extra

"""
import gymnasium as gym
import numpy as np
import random
import sapien
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import time
    
import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.structs import Pose
        
import mani_skill.examples.clevr_env  # do import to register env, not used otherwise
from mani_skill.examples.clevr_env_solver import get_grasp_pose_and_obb
from mani_skill.examples.utils_env_interventions import move_object_onto
from mani_skill.examples.utils_trajectory import generate_curve_torch
from mani_skill.examples.utils_traj_tokens import to_prefix_suffix
from mani_skill.examples.utils_traj_tokens import getActionEncDecFunction
from utils_trajectory import DummyCamera

from pdb import set_trace

RAND_MAX = 2**32 - 1
SAVE_FREQ = 10
SAVE_VIDEO = False


def getMotionPlanner(env):
    if env.unwrapped.robot_uids in ("panda", "panda_wristcam"):
        from mani_skill.examples.motionplanning.panda.motionplanner import \
            PandaArmMotionPlanningSolver as RobotArmMotionPlanningSolver
    elif env.unwrapped.robot_uids == "fetch":
        from mani_skill.examples.motionplanning.fetch.motionplanner import \
            FetchArmMotionPlanningSolver as RobotArmMotionPlanningSolver
    else:
        raise ValueError(f"no motion planner for {env.unwrapped.robot_uids}")
    return RobotArmMotionPlanningSolver



@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "ClevrMove-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb+depth+segmentation"
    """Observation mode"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: int = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
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

    seed: Annotated[Optional[Union[int, List[int], str]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

    run_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-m"])] = "script"
    """Run a script|interactive|first. first renders first frame only"""

    action_encoder: Annotated[Optional[str], tyro.conf.arg(aliases=["-ae"])] = "xyzrotvec-cam-proj2"
    """Action encoding"""

    object_dataset: Annotated[Optional[str], tyro.conf.arg(aliases=["-od"])] = "clevr"
    """Dataset from which we sample objects"""


def reset_random(args, orig_seeds):
    if orig_seeds is None:
        seed = random.randrange(RAND_MAX)
    elif isinstance(orig_seeds, list):
        seed = orig_seeds.pop()
    elif isinstance(orig_seeds, int):
        seed = orig_seeds        
    else:
        raise ValueError
    args.seed = [seed]
    np.random.seed(seed)

    
def iterate_env(args: Args, vis=True, model=None, max_iter=10**6):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False

    env = gym.make(
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
        robot_uids="panda",  #fetch, panda_wristcam
        scene_dataset="Table", # Table, ProcTHOR
        object_dataset=args.object_dataset, # clevr, ycb, objaverse
        # **args.env_kwargs
    )

    if args.record_dir:
        env = RecordEpisode(env, args.record_dir, info_on_video=False, 
                            save_trajectory=True, max_steps_per_video=env._max_episode_steps,
                            save_on_reset=SAVE_FREQ == 1,
                            record_env_state=True)
    
    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)
        print("Render mode", args.render_mode)
        print("Obs mode", args.obs_mode)

    filter_visible = True
    enc_func, dec_func = getActionEncDecFunction(args.action_encoder)
    print("filter visible objects", filter_visible)
    print("action encoder", args.action_encoder)

    orig_seeds = args.seed
    for i in range(max_iter):
        reset_random(args, orig_seeds)
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

        # Note: overwriting _default_sensor_configs in env with camera called "render_camera"

        env_idx = 0
        # get image before intervention (randomization of human_render_camera is done by clevr_env.py)
        
        # #-----
        # # Warning, taking an image form obs/rendering it results in different calibrations!
        # #----
        # images = env.base_env.scene.get_human_render_camera_images('render_camera')
        # image_before = images['render_camera'][env_idx].numpy()
        # #image_before = obs['sensor_data']['render_camera']['rgb'][0].detach().numpy()
        # assert image_before.shape == (448, 448, 3)

        # # do intervention (this will set env.base_env.cubeA onto cubeB)
        # obj_start, obj_end, action_text = move_object_onto(env, pretend=True)
        # env.unwrapped.set_goal_pose(obj_end)
        
        #camera = env.base_env.scene.human_render_cameras['render_camera'].camera
        # grasp_pose, _ = get_grasp_pose_and_obb(env.base_env)
        # grasp_pose = Pose.create_from_pq(p=grasp_pose.get_p(), q=grasp_pose.get_q())
        # tcp_pose = env.unwrapped.agent.tcp.pose
        # robot_pose = env.base_env.agent.robot.get_root_pose()

        obj_start = Pose(obs["extra"]["obj_start"].clone().detach())
        obj_end = Pose(obs["extra"]["obj_end"].clone().detach())
        grasp_pose = Pose(obs["extra"]["grasp_pose"].clone().detach())
        tcp_pose = Pose(obs["extra"]["tcp_pose"].clone().detach())
        robot_pose = Pose(obs["extra"]["robot_pose"].clone().detach())

        try:
            camera_intrinsic = obs["sensor_param"]["render_camera"]["intrinsic_cv"].clone().detach()
            camera_extrinsic = obs["sensor_param"]["render_camera"]["extrinsic_cv"].clone().detach()
            image_before = obs["sensor_data"]["render_camera"]["rgb"][0].clone().detach()
            width, height, _ = image_before.shape
            camera = DummyCamera(camera_intrinsic, camera_extrinsic, width, height)
        except KeyError:
            image_before = None
            camera = env.base_env.scene.human_render_cameras['render_camera'].camera

        action_text = None
        for x in obs["extra"].keys():
            if x.startswith("action_text_"):
                action_text = str(x).replace("action_text_", "")
        assert action_text is not None

        #enc_func, dec_func = getEncDecFunc("xyzrotvec-rbt")
        prefix, token_str, curve_3d, orns_3d, info = to_prefix_suffix(obj_start, obj_end,
                                                                      camera, grasp_pose, tcp_pose,
                                                                      action_text, enc_func, robot_pose=robot_pose)
        # skip saving this file, we won't use it anyway
        if filter_visible and info["didclip_traj"]:
            continue

        json_dict = dict(prefix=prefix, suffix=token_str,
                         action_text=action_text,
                         camera_extrinsic=camera.get_extrinsic_matrix().detach().numpy().tolist(),
                         camera_intrinsic=camera.get_intrinsic_matrix().detach().numpy().tolist(),
                         obj_start_pose=obj_start.raw_pose.detach().numpy().tolist(),
                         obj_end_pose=obj_end.raw_pose.detach().numpy().tolist(),
                         robot_pose=robot_pose.raw_pose.detach().numpy().tolist(),
                         tcp_start_pose=tcp_pose.raw_pose.detach().numpy().tolist(),
                         grasp_pose=grasp_pose.raw_pose.detach().numpy().tolist(),
                         info=info,
                         seed=args.seed[0])
        
        encode_decode_trajectory = True
        if encode_decode_trajectory:
            curve_3d_est, orns_3d_est = dec_func(token_str, camera, robot_pose=robot_pose)
            curve_3d = curve_3d_est  # set the unparsed trajectory one used for policy
            orns_3d = orns_3d_est
            enc_func, dec_func = getActionEncDecFunction(args.action_encoder)

        # Evaluate the trajectory
        if args.run_mode == "script" or model:
            assert args.control_mode == "pd_joint_pos"
            if verbose and info["didclip_traj"]:
                print("Warning refered object out of camera view.")
                
            if model:
                img_out, text, label, token_pred = model.make_predictions(image_before, prefix)
                json_dict["prediction"] = token_pred
                curve_3d_pred, orns_3d_pred = dec_func(token_pred, camera=camera, robot_pose=robot_pose)
                curve_3d = curve_3d_pred  # set the unparsed trajectory one used for policy
                orns_3d = orns_3d_pred

            # convert two keypoints into motion sequence
            assert curve_3d.shape[1] == 2 and orns_3d.shape[1] == 2  # start and stop poses
            _, curve_3d_i = generate_curve_torch(curve_3d[:, 0], curve_3d[:, -1], num_points=3)
            grasp_pose = Pose.create_from_pq(p=curve_3d[:, 0], q=orns_3d[:, 0])
            reach_pose = grasp_pose * sapien.Pose([0, 0, -0.10]) # Go above the object before grasping
            lift_pose = Pose.create_from_pq(p=curve_3d_i[:, 1], q=orns_3d[:, 1])
            align_pose = Pose.create_from_pq(p=curve_3d_i[:, 2], q=orns_3d[:, 1])
            pre_align_pose = align_pose * sapien.Pose([0, 0, -0.10]) # Go above before dropping

            # execute motion sequence using IK solver
            RobotArmMotionPlanningSolver = getMotionPlanner(env)
            planner = RobotArmMotionPlanningSolver(
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
            planner.move_to_pose_with_screw(pre_align_pose)
            planner.move_to_pose_with_screw(align_pose)
            planner.open_gripper()
            final_reward = env.unwrapped.eval_reward()[0]
            planner.close()
            json_dict["reward"] = float(final_reward)
            if verbose:
                print(f"reward {final_reward:0.2f} seed", args.seed[0])

        elif args.run_mode == "interactive":
            run_interactive(env)
        elif args.run_mode == "first":
            # only render first frame
            pass
        else:
            raise ValueError

        if args.record_dir:
            try:
                seg = env._trajectory_buffer.observation['sensor_data']['render_camera']['segmentation']
                if seg.max() <= 255:
                    seg = seg.astype(np.uint8)
                env._trajectory_buffer.observation['sensor_data']['render_camera']['segmentation'] = seg
            except KeyError:
                pass

            #if i % SAVE_FREQ == 0:
            # keep the transition from reset (which does not have an action)
            env.flush_trajectory(save=True, ignore_empty_transition=False)
            if SAVE_VIDEO:
                video_name =  f"CLEVR_{str(args.seed[0]).zfill(10)}"
                env.flush_video(name=video_name, save=True)

        yield image_before, json_dict, args.seed[0]

    env.close()


def run_interactive(env):
    env.print_sim_details()

    print("Entering do nothing loop.")
    while True:
        time.sleep(.1)
        env.base_env.render_human()


# def save_dataset(sample_generator, N: int, dataset_path):
#     dataset_orig = dataset_path
#     dataset_path = dataset_path / "dataset"
#     os.makedirs(dataset_path, exist_ok=True)

#     # Initialize a ThreadPoolExecutor with one or more workers
#     executor = ThreadPoolExecutor(max_workers=3)
#     def save_image(image, path) -> None:
#         Image.fromarray(image).save(path, format='JPEG')
        
#     def save_image_async(image, path) -> None:
#         # Submit the save_image task to be run in a separate thread
#         executor.submit(save_image, image, path)

#     def save_labels(annotations):
#         json_file = dataset_orig / "_annotations.all.jsonl"
#         # "a" means append to the file
#         with open(json_file, "a") as f:
#             for obj in annotations:
#                 json.dump(obj, f)
#                 f.write("\n")

#     def get_num_lines():
#         json_file = dataset_orig / "_annotations.all.jsonl"
#         if Path(json_file).exists():
#             with open(json_file) as f:
#                 return sum(1 for line in f)
#         else:
#             return 0
        
#     #N_cur = get_num_lines()
#     N_cur = 0
#     N_remaining = N - N_cur
#     annotations = []
#     for i in tqdm(range(N_remaining),total=N_remaining):
#         image_before, json_dict, rnd_seed = next(sample_generator)
#         sample_name = str(rnd_seed).zfill(10)
#         if image_before is not None:
#             image_filename = f"CLEVR_{sample_name}.jpg"
#             json_dict["image"] = image_filename
#             save_image_async(image_before, dataset_path / image_filename)
#         annotations.append(json_dict)
#         if i % 100 == 0:
#             save_labels(annotations)
#             annotations = []
#     save_labels(annotations)
#     print("done.")

import asyncio
import time
from copy import deepcopy
import multiprocessing

async def async_run_iteration(parsed_args, N_samples, process_num=None):
    """Runs the environment iteration asynchronously."""
    desc = "Processing Iterations"
    if process_num is not None:
        desc += f"-{process_num}"
    env_iter = iterate_env(parsed_args, vis=False, max_iter=N_samples)
    loop = asyncio.get_running_loop()
    for _ in tqdm(range(N_samples), desc=desc):
        await loop.run_in_executor(None, lambda: next(env_iter))

def run_iteration(parsed_args, N_samples, process_num=None, progress_bar=None):
    """Runs the environment iteration in a separate process."""
    env_iter = iterate_env(parsed_args, vis=False, max_iter=N_samples)
    for _ in range(N_samples):
        _ = next(env_iter)
        if progress_bar is not None:
            progress_bar.value += 1

def save_multiproces(parsed_args, N_samples, N_processes=10):
        parsed_args.run_mode = "first"
        dataset_path = parsed_args.record_dir

        if isinstance(parsed_args.seed, int):
            assert N_processes == 10
            rng = np.random.default_rng(parsed_args.seed)
            parsed_args.seed = rng.integers(0, RAND_MAX, N_samples).tolist()
        
        if N_processes == 1:
            dataset_path = Path(dataset_path)
            os.makedirs(dataset_path, exist_ok=True)
            env_iter = iterate_env(parsed_args, vis=False, max_iter=N_samples)
            for _ in tqdm(range(N_samples)):
                _ = next(env_iter)
        else:
            samples_per_process = N_samples // N_processes
            tasks = []
            
            progress_bar = multiprocessing.Value("i", 0)  
            for p_num in range(N_processes):
                dataset_path_p = Path(dataset_path) / f"p{p_num}"
                os.makedirs(dataset_path_p, exist_ok=True)
                args_copy = deepcopy(parsed_args)
                args_copy.record_dir = dataset_path_p
                #tasks.append(async_run_iteration(args_copy, N_samples=samples_per_process, process_num=p_num))
                p = multiprocessing.Process(target=run_iteration, args=(args_copy, samples_per_process, p_num, progress_bar), name=f"Worker-{p_num+1}")
                tasks.append(p)
                p.start()
                time.sleep(1.1)

            # Display tqdm progress in the main process
            with tqdm(total=N_samples, desc="Total Progress", position=0, leave=True) as pbar:
                last_count = 0
                while any(p.is_alive() for p in tasks):  # Update while processes are running
                    current_count = progress_bar.value
                    pbar.update(current_count - last_count)  # Update tqdm only for new progress
                    last_count = current_count
                    time.sleep(1)  # Prevents excessive updates
            
            #await asyncio.gather(*tasks)
            for p in tasks:
                p.join()  # Wait for all processes to finish

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    dataset_path = parsed_args.record_dir

    if isinstance(parsed_args.seed, str):
        import json
        with open(parsed_args.seed, "r") as f_obj:
            seeds = json.load(f_obj)
            parsed_args.seed = seeds

    if dataset_path is None:
        env_iter = iterate_env(parsed_args, vis=True)
        while True:
            _ = next(env_iter)
    else:
        N_samples = int(140_000+10_000)
        #asyncio.run(save_multiproces(parsed_args, N_samples))
        save_multiproces(parsed_args, N_samples)

        # N_samples = int(2/0.8)
        #N_samples = 50
        # parsed_args.run_mode = "first"
        # if isinstance(parsed_args.seed, int):
        #     rng = np.random.default_rng(parsed_args.seed)
        #     parsed_args.seed = rng.integers(0, RAND_MAX, N_samples).tolist()
        # dataset_path = Path(dataset_path)
        # os.makedirs(dataset_path, exist_ok=True)
        # env_iter = iterate_env(parsed_args, vis=False, max_iter=N_samples)
        # for _ in tqdm(range(N_samples)):
        #     _ = next(env_iter)
        # # save_dataset(iterate_env(parsed_args, vis=False, max_iter=N_samples), N=int(N_samples), dataset_path=dataset_path)
        