import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from cvla.utils_trajectory import project_point
import torch
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
from matplotlib import pyplot as plt
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


def draw_multiple_poses_on_image(image, poses, camera, radius=5, axis_length=0.05):
    """
    Draws multiple 7D poses on an image and visualizes the local coordinate axes (x, y, z).

    Args:
        image (np.ndarray): The input RGB image of shape (H, W, 3).
        poses (List[torch.Tensor or np.ndarray]): A list of 7D poses (x, y, z, qx, qy, qz, qw).
        camera: A camera object with get_intrinsic_matrix() and get_extrinsic_matrix() methods.
        radius (int): Radius of the circle drawn at the pose origin.
        axis_length (float): Length of each axis arrow in 3D space (in meters).

    Returns:
        np.ndarray: The image with visualized poses and coordinate axes.
    """
    img = image.copy()

    positions = []
    x_dirs = []
    y_dirs = []
    z_dirs = []

    for pose in poses:
        if isinstance(pose, np.ndarray):
            pose = torch.tensor(pose)
        position = pose[:3]
        quat = pose[3:]

        # Convert quaternion to rotation matrix
        rot = R.from_quat(quat.cpu().numpy())
        rot_matrix = torch.tensor(rot.as_matrix(), dtype=torch.float32)  # shape: (3, 3)

        # Extract x, y, z axis directions from the rotation matrix
        x_axis = rot_matrix[:, 0]  # x-direction
        y_axis = rot_matrix[:, 1]  # y-direction
        z_axis = rot_matrix[:, 2]  # z-direction

        positions.append(position)
        x_dirs.append(x_axis)
        y_dirs.append(y_axis)
        z_dirs.append(z_axis)

    positions = torch.stack(positions)   # (N, 3)
    x_dirs = torch.stack(x_dirs)         # (N, 3)
    y_dirs = torch.stack(y_dirs)
    z_dirs = torch.stack(z_dirs)

    # Compute the 3D endpoints of the axes
    x_ends = positions + axis_length * x_dirs
    y_ends = positions + axis_length * y_dirs
    z_ends = positions + axis_length * z_dirs

    # Project the origins and endpoints to 2D image coordinates
    pts_start = project_point(camera, positions)  # (N, 2)
    x_2d = project_point(camera, x_ends)
    y_2d = project_point(camera, y_ends)
    z_2d = project_point(camera, z_ends)

    for i in range(len(poses)):
        x0, y0 = int(pts_start[i, 0]), int(pts_start[i, 1])
        cv2.circle(img, (x0, y0), radius, (255, 255, 255), -1)  # Draw the origin as a white dot

        # Helper function to draw arrows
        def draw_arrow(x1, y1, x2, y2, color):
            cv2.arrowedLine(img, (x1, y1), (x2, y2), color, 2, tipLength=0.2)

        # Draw the coordinate axes
        draw_arrow(x0, y0, int(x_2d[i, 0]), int(x_2d[i, 1]), (0, 0, 255))   # X-axis: red
        draw_arrow(x0, y0, int(y_2d[i, 0]), int(y_2d[i, 1]), (0, 255, 0))   # Y-axis: green
        draw_arrow(x0, y0, int(z_2d[i, 0]), int(z_2d[i, 1]), (255, 0, 0))   # Z-axis: blue

        # Label the origin point
        cv2.putText(img, f'P{i}', (x0 + 5, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img

#for temperoray visualization
hand_cam_view = []
virtual_cam_view = []
depth_cam_view = []
cam_pose = Pose.create_from_pq(p=[0,0,0],q=[1,0,0,0])


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


def reset_random(args, orig_seeds=None):
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


def iterate_env(args: Args, vis=True, model=None):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud", "top_view"]:
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

    orig_seeds = args.seed
    N_valid_samples = 0
    max_attempts = 10**6
    for i in range(max_attempts):
        #here if set parameters as (agrs, orig_seeds) can same scenes
        reset_random(args)
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
        elif "top" in str(args.camera_views): # from the top-view above the fisrt object, can't see two objects
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
            #camera_intrinsic[0,:-1,2] = 0
            camera_extrinsic = obs["sensor_param"]["render_camera"]["extrinsic_cv"].clone().detach()
            image_before = obs["sensor_data"]["render_camera"]["rgb"][0].clone().detach()
            image_exp = obs["sensor_data"]["render_camera"]["rgb"][0].clone().detach()
            depth = obs["sensor_data"]["render_camera"]["depth"][0].clone().detach()
            width, height, _ = image_before.shape
            camera = DummyCamera(camera_intrinsic, camera_extrinsic, width, height)
            # add depth to image_before if this mode take depth
            if "depth" in str(args.obs_mode) and "top" not in str(args.camera_views):
                image_before = (depth, image_before)
        except KeyError:
            image_before = obs["sensor_data"]["render_camera"]["rgb"][0].clone().detach()
            camera = env.base_env.scene.human_render_cameras['render_camera'].camera

        action_text = env.unwrapped.get_obs_scene()["text"]
        assert isinstance(action_text, str) and action_text not in (None, ""), f"action_text: {action_text}"
        
        prefix, token_str, curve_3d, orns_3d, info = to_prefix_suffix(obj_start, obj_end,
                                                                      camera, grasp_pose, tcp_pose,
                                                                      action_text, enc_func, robot_pose=robot_pose)
           
        '''
        extra:
        obj_start
        obj_end
        grasp_pose
        tcp_pose
        robot_pose
            
        '''
        

        #top_view.append(image_before)
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
        def get_pose_of_new_predicts(image,prefix,model):
            pose_list = []
            return pose_list

        # Evaluate the trajectory
        if args.run_mode == "script" or model:
            assert args.control_mode == "pd_joint_pos"
            if verbose and info["didclip_traj"]:
                print("Warning refered object out of camera view.")
                

            if model:
                '''
                print(f"prefix:{prefix}")
                print(f"image type:{type(image_before)}")
                print("visualize of image before")
                show_before = image_before.cpu().numpy()
                plt.imshow(show_before)
                plt.axis('off')  # Turn off axis numbers/labels
                plt.show()
                '''
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

            #debug only
            #env.render()
            #env.scene._sapien_update_render(update_sensors=True)

            #check the actual location of robot
            _ = env.base_env.get_obs()
            obs2 = env.base_env.get_obs()
            cam_pose = env.render_camera_config.pose
            #draw these two points on the imgage
            image1 = obs2["sensor_data"]["render_camera"]["rgb"][0].clone().detach()
            image1 = image1.cpu().numpy()
            #pose_raw2 = env.render_camera_config.pose
            #pose_raw2 = pose_raw2.raw_pose.squeeze(0)  # make it a tensor
            #point_3d = [obs2["extra"]["tcp_pose"][0], pose_raw2]
            obj_start = obs2["extra"]["obj_start"][0]
            obj_end = obs2["extra"]["obj_end"][0]
            zero_t = torch.zeros_like(obj_start)
            zero_t[3:] = obj_start[3:]
            cam_pose_p = cam_pose.p
            cam_pose_q = cam_pose.q
            low_cam = torch.tensor([cam_pose_p[0][0], cam_pose_p[0][1], 0, cam_pose_q[0][0], 
                                     cam_pose_q[0][1], cam_pose_q[0][2], cam_pose_q[0][3]], dtype=torch.float32)
            
            point_3d = [obj_start, obj_end,low_cam ]
            output_img = draw_multiple_poses_on_image(image1, point_3d, camera)            
            plt.imshow(output_img)
            plt.axis('off')
            plt.show()
            
            image2 = obs2["sensor_data"]["top_camera"]["rgb"][0].clone().detach()
            image2 = image2.cpu().numpy()
            cam_pose = env.top_camera_config.pose
            cam_pose_p = cam_pose.p
            cam_pose_q = cam_pose.q
            cam2 = torch.tensor([cam_pose_p[0][0], cam_pose_p[0][1], 0, cam_pose_q[0][0], 
                                     cam_pose_q[0][1], cam_pose_q[0][2], cam_pose_q[0][3]], dtype=torch.float32)
            # parameter
            camera_intrinsic = obs["sensor_param"]["top_camera"]["intrinsic_cv"].clone().detach()
            #camera_intrinsic[0,:-1,2] = 0
            camera_extrinsic = obs["sensor_param"]["top_camera"]["extrinsic_cv"].clone().detach()
            image_before1 = obs["sensor_data"]["top_camera"]["rgb"][0].clone().detach()
            width, height, _ = image_before1.shape
            camera2 = DummyCamera(camera_intrinsic, camera_extrinsic, width, height)
            
            output_img = draw_multiple_poses_on_image(image2, point_3d, camera2)
            #image3 = obs1["sensor_data"]["render_camera"]["depth"][0].clone().detach()
            #depth_cam_view.append(image3)
            
            #for key in obs1["sensor_data"]["render_camera"].keys():
            #   print(key)
            plt.imshow(output_img)
            plt.axis('off')
            plt.show()

            

            
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
           
            
            #get the current observation from top
            
            obs1 = env.base_env.get_obs() 
            image1 = obs1["sensor_data"]["render_camera"]["rgb"][0].clone().detach()
            virtual_cam_view.append(image1)
            
            
            #plt.axis('off')  # Turn off axis numbers/labels
            #plt.show()
            
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

        # adding the top view image to 
        if "top" in str(args.camera_views):
            image_before = image_before
            #obs["sensor_data"]["render_camera"]["depth"][0] = obs1["sensor_data"]["hand_camera"]["rgb"][0].clone().detach()

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