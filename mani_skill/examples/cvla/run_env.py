"""
Main entry point for running the CVLA environment. See readme for details.
"""
import os
import json
import time
import random
import traceback
import multiprocessing
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

import tyro
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import sapien

from mani_skill.utils.structs import Pose
from mani_skill.utils.wrappers import RecordEpisode
import mani_skill.examples.cvla.cvla_env  # do import to register env, not used otherwise
from mani_skill.examples.cvla.utils_trajectory import generate_curve_torch, DummyCamera
from mani_skill.examples.cvla.utils_traj_tokens import getActionEncInstance, to_prefix_suffix
from mani_skill.examples.cvla.utils_record import apply_check_object_pixels_obs
from mani_skill.examples.cvla.utils_record import downcast_seg_array

import gc
import torch


RAND_MAX = 2**32 - 1
SAVE_FREQ = 1  # save after every reset
RESET_HARD = 10  # re-start environment after every n steps
SAVE_VIDEO = False  # save videos
# minimum percentage of image that must be object, set to None to disable checking
MIN_OBJ_VISIBLE_PERCENT = 0.5


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
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "CvlaMove-v1"
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
    """Run mode, options are script, interactive, first"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = "panda"
    """Robots, options are: panda, panda_wristcam, xarm6_robotiq, floating_inspire_hand_right"""

    scene_dataset: Annotated[Optional[str], tyro.conf.arg(aliases=["-sd"])] = "Table"
    """Scene datasets: options are: Table, ProcTHOR"""

    scene_options: Annotated[Optional[str], tyro.conf.arg(aliases=["-so"])] = "fixed"
    """Randomize the scene"""

    object_dataset: Annotated[Optional[str], tyro.conf.arg(aliases=["-od"])] = "clevr"
    """Dataset from which we sample objects, options are: clevr, ycb, objaverse"""

    camera_views: Annotated[Optional[str], tyro.conf.arg(aliases=["-cv"])] = "random_side"
    """Dataset from which we sample objects"""

    action_encoder: Annotated[Optional[str], tyro.conf.arg(aliases=["-ae"])] = "xyzrotvec-cam-1024xy"
    """Action encoding"""

    N_samples: Annotated[Optional[int], tyro.conf.arg(aliases=["-N"])] = 50
    """Number of samples"""


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


args: Args, vis=True, model=None):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False

    # define make env as a function to enable hard resets
    def make_env():
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
            robot_uids=args.robot_uids,
            scene_dataset=args.scene_dataset,
            object_dataset=args.object_dataset,
            camera_views=args.camera_views,
            scene_options=args.scene_options,
            # camera_cfgs={"use_stereo_depth": True, },
            # **args.env_kwargs
        )
        if args.record_dir:
            env = RecordEpisode(env, args.record_dir, info_on_video=False,
                                save_trajectory=True, max_steps_per_video=env._max_episode_steps,
                                save_on_reset=SAVE_FREQ == 1,
                                record_env_state=True)
        return env

    env = make_env()

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)
        print("Render mode", args.render_mode)
        print("Obs mode", args.obs_mode)

    filter_visible = True
    action_encoder = getActionEncInstance(args.action_encoder)
    enc_func, dec_func = action_encoder.encode_trajectory, action_encoder.decode_trajectory

    print("action encoder", args.action_encoder)
    print("filter visible objects", filter_visible)

    orig_seeds = args.seed
    N_valid_samples = 0
    max_attempts = 10**6
    for i in range(max_attempts):
        reset_random(args, orig_seeds)
        assert isinstance(args.seed, list)

        if i != 0 and i % RESET_HARD == 0:
            del env
            env = make_env()
        try:
            obs, _ = env.reset(seed=args.seed[0], options=dict(reconfigure=True))
        except Exception as e:  # Catch all exceptions, including AssertionError
            print(f"Encountered error {e.__class__.__name__} at seed {args.seed[0]} while resetting env. Skipping this iteration.")
            print(e)
            traceback.print_exc()  # Prints the full traceback
            gc.collect()
            torch.cuda.empty_cache()
            continue

        if MIN_OBJ_VISIBLE_PERCENT is None:
            obj_are_vis = True
        else:
            obj_are_vis = apply_check_object_pixels_obs(obs, env, N_percent=MIN_OBJ_VISIBLE_PERCENT)
        if not obj_are_vis:
            print("Warning: object not visible, skipping sample")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Note: when using RecordEpisode this will create 20x the number of saved frames
        # so 75GB -> 1.5 TB, which is no good.
        # Let the objects settle (!)
        # for _ in range(20):
        #    _ = env.step(obs["agent"]["qpos"][..., :8])

        if args.seed is not None:
            env.action_space.seed(args.seed[0])
        if vis and args.render_mode is not None:
            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = args.pause
            env.render()
        else:
            env.render()

        # Not parrelized
        # env_idx = 0

        # -----
        # Warning, taking an image form obs/rendering it results in different calibrations!
        # e.g. images = env.base_env.scene.get_human_render_camera_images('render_camera')
        # -----
        obj_start = Pose(obs["extra"]["obj_start"].clone().detach())
        obj_end = Pose(obs["extra"]["obj_end"].clone().detach())
        grasp_pose = Pose(obs["extra"]["grasp_pose"].clone().detach())
        tcp_pose = Pose(obs["extra"]["tcp_pose"].clone().detach())
        robot_pose = Pose(obs["extra"]["robot_pose"].clone().detach())

        try:
            camera_intrinsic = obs["sensor_param"]["render_camera"]["intrinsic_cv"].clone().detach()
            camera_extrinsic = obs["sensor_param"]["render_camera"]["extrinsic_cv"].clone().detach()
            image_before = obs["sensor_data"]["render_camera"]["rgb"][0].clone().detach()
            depth = obs["sensor_data"]["render_camera"]["depth"][0].clone().detach()
            width, height, _ = image_before.shape
            camera = DummyCamera(camera_intrinsic, camera_extrinsic, width, height)
            # add depth to image_before
            image_before = (depth, image_before)
        except KeyError:
            image_before = None
            camera = env.base_env.scene.human_render_cameras['render_camera'].camera

        action_text = env.unwrapped.get_obs_scene()["text"]
        assert isinstance(action_text, str) and action_text not in (None, ""), f"action_text: {action_text}"

        prefix, token_str, curve_3d, orns_3d, info = to_prefix_suffix(obj_start, obj_end,
                                                                      camera, grasp_pose, tcp_pose,
                                                                      action_text, enc_func, robot_pose=robot_pose)

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
                         seed=args.seed[0],
                         iter_reached=i,
                         )

        encode_decode_trajectory = True
        if encode_decode_trajectory:
            curve_3d_est, orns_3d_est = dec_func(token_str, camera, robot_pose=robot_pose)
            curve_3d = curve_3d_est  # set the unparsed trajectory one used for policy
            orns_3d = orns_3d_est

        # Evaluate the trajectory
        if args.run_mode == "script" or model:
            assert args.control_mode == "pd_joint_pos"
            if verbose and info["didclip_traj"]:
                print("Warning refered object out of camera view.")

            if model:
                _, _, _, token_pred = model.make_predictions(image_before, prefix)
                json_dict["prediction"] = token_pred
                if token_pred == "" or token_pred is None:
                    print("Warning: empty prediction, failing")
                    json_dict["reward"] = 0
                    gc.collect()
                    torch.cuda.empty_cache()
                    yield image_before, json_dict, args.seed[0]
                    continue

                try:
                    curve_3d_pred, orns_3d_pred = dec_func(token_pred, camera=camera, robot_pose=robot_pose)
                    curve_3d = curve_3d_pred  # set the unparsed trajectory one used for policy
                    orns_3d = orns_3d_pred
                # TODO(max): this should only catch value errors
                except:
                    print("Warning: exception during decoding tokens, failing", token_pred)
                    json_dict["reward"] = 0
                    gc.collect()
                    torch.cuda.empty_cache()
                    yield image_before, json_dict, args.seed[0]
                    continue

            # start and stop poses
            if curve_3d.shape[1] != 2 or orns_3d.shape[1] != 2:
                print("Warning: Model decoded something that is not a valid trajectory")
                json_dict["reward"] = 0.0
                gc.collect()
                torch.cuda.empty_cache()
                yield image_before, json_dict, args.seed[0]
                N_valid_samples += 1
                continue

            # convert two keypoints into motion sequence
            _, curve_3d_i = generate_curve_torch(curve_3d[:, 0], curve_3d[:, -1], num_points=3)
            grasp_pose = Pose.create_from_pq(p=curve_3d[:, 0], q=orns_3d[:, 0])
            reach_pose = grasp_pose * sapien.Pose([0, 0, -0.10])  # Go above the object before grasping
            lift_pose = Pose.create_from_pq(p=curve_3d_i[:, 1], q=orns_3d[:, 1])
            align_pose = Pose.create_from_pq(p=curve_3d_i[:, 2], q=orns_3d[:, 1])
            pre_align_pose = align_pose * sapien.Pose([0, 0, -0.10])  # Go above before dropping

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
            # run_interactive(env)
            planner.close_gripper()
            planner.move_to_pose_with_screw(lift_pose)
            planner.move_to_pose_with_screw(pre_align_pose)
            planner.move_to_pose_with_screw(align_pose)
            # run_interactive(env)
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
            # if i % SAVE_FREQ == 0:
            # keep the transition from reset (which does not have an action)

            downcast_seg_array(env)
            env.flush_trajectory(save=True, ignore_empty_transition=False)
            # to skip saving do: env.flush_trajectory(save=False)

            if SAVE_VIDEO:
                video_name = f"CLEVR_{str(args.seed[0]).zfill(10)}"
                env.flush_video(name=video_name, save=True)

        del obs
        gc.collect()
        torch.cuda.empty_cache()
        yield image_before, json_dict, args.seed[0]

        N_valid_samples += 1

    env.close()


def run_interactive(env):
    env.print_sim_details()
    print("Entering do nothing loop: Ctrl-C to continue")
    try:
        while True:
            time.sleep(.1)
            env.base_env.render_human()
    except KeyboardInterrupt:
        print("\nCtrl+C detected, continuing.")


def run_iteration(parsed_args, N_samples, process_num=None, progress_bar=None):
    """Runs the environment iteration in a separate process."""
    env_iter = iterate_env(parsed_args, vis=False)
    for _ in range(N_samples):
        _ = next(env_iter)
        if progress_bar is not None:
            progress_bar.value += 1


def save_multiproces(parsed_args, N_samples, N_processes=10):
    from mani_skill.examples.cvla.utils_record import check_no_uncommitted_changes, get_git_commit_hash
    parsed_args.run_mode = "first"
    dataset_path = Path(parsed_args.record_dir)
    os.makedirs(dataset_path, exist_ok=True)

    # save command line arguments in nice format
    if N_samples > 100:
        check_no_uncommitted_changes()
    commit_hash = get_git_commit_hash()
    with open(dataset_path / "args.txt", "w") as f:
        f.write(f"git_commit: {commit_hash}\n")
        for arg in vars(parsed_args):
            f.write(f"{arg}: {getattr(parsed_args, arg)}\n")

    # set random seeds, be careful to not copy same seeds between processes
    if N_processes > 1:
        assert parsed_args.seed is None
    if isinstance(parsed_args.seed, int):
        assert N_processes == 1
        rng = np.random.default_rng(parsed_args.seed)
        parsed_args.seed = rng.integers(0, RAND_MAX, N_samples).tolist()

    # don't multiprocess
    if N_processes == 1:
        # don't set N_samples in iterate_env, so that e.g. re-generate can work for visibility
        env_iter = iterate_env(parsed_args, vis=False)
        for _ in tqdm(range(N_samples)):
            try:
                _ = next(env_iter)
            except StopIteration:
                break
    else:
        samples_per_process = N_samples // N_processes
        progress_bar = multiprocessing.Value("i", 0)

        tasks = []
        for p_num in range(N_processes):
            dataset_path_p = Path(dataset_path) / f"p{p_num}"
            os.makedirs(dataset_path_p, exist_ok=True)
            args_copy = deepcopy(parsed_args)
            args_copy.record_dir = dataset_path_p
            p = multiprocessing.Process(target=run_iteration, args=(args_copy, samples_per_process, p_num, progress_bar), name=f"Worker-{p_num+1}")
            tasks.append(p)
            p.start()
            time.sleep(1.1)  # Give some time for processes to start

        # Display tqdm progress in the main process
        with tqdm(total=N_samples, desc="Total Progress", position=0, leave=True) as pbar:
            last_count = 0
            while any(p.is_alive() for p in tasks):  # Update while processes are running
                current_count = progress_bar.value
                pbar.update(current_count - last_count)  # Update tqdm only for new progress
                last_count = current_count
                time.sleep(1)  # Prevents excessive updates

        # await asyncio.gather(*tasks)
        for p in tasks:
            p.join()  # Wait for all processes to finish


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    dataset_path = parsed_args.record_dir

    if isinstance(parsed_args.seed, str):
        with open(parsed_args.seed, "r") as f_obj:
            seeds = json.load(f_obj)
            parsed_args.seed = seeds

    if dataset_path is None:  # Normal run
        env_iter = iterate_env(parsed_args, vis=True)
        while True:
            _ = next(env_iter)
    else:
        # asyncio.run(save_multiproces(parsed_args, N_samples))
        N_processes = 1
        if parsed_args.N_samples > 100:
            if parsed_args.object_dataset == "clevr":
                N_processes = 10
            else:
                N_processes = 5

        save_multiproces(parsed_args, parsed_args.N_samples, N_processes=N_processes)
