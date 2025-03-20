# see examples/motionplanning/panda/solutions/stack_cube.py for the template of this
import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import StackCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from scipy.spatial.transform import Rotation as R

def object_is_rotationally_invariant(obj):
    if obj.name.startswith("sphere"):
        return True
    return False

def get_grasp_pose_and_obb(env: StackCubeEnv):
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    obb = get_actor_obb(env.cubeA)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    if object_is_rotationally_invariant(env.cubeA):
        # TODO(max): this is bad and can cause problems, only consider z-axis
        # keep this simple for now
        grasp_pose.set_q(env.agent.tcp.pose.get_q()[0])
        #rot_initial = R.from_quat(env.agent.tcp.pose.get_q()[0].numpy(), scalar_first=True)
        #rot_initial_z = rot_initial.as_euler('xyz', degrees=True)[2]
        #rot_grasp = R.from_quat(grasp_pose.get_q(), scalar_first=True)
        #rot_grasp_xy = rot_grasp.as_euler('xyz', degrees=True)[:2].tolist()
        #rot_grasp_new = R.from_euler('xyz', rot_grasp_xy + [rot_initial_z], degrees=True) 
        #json_dict['traj_q'][0] = rot_grasp_new.as_quat(scalar_first=True).tolist()  # json serializable


    # Search a valid pose
    # angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    # angles = np.repeat(angles, 2)
    # angles[1::2] *= -1
    # for angle in angles:
    #     delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
    #     grasp_pose2 = grasp_pose * delta_pose
    #     res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
    #     if res == -1:
    #         continue
    #     grasp_pose = grasp_pose2
    #     break
    # else:
    #     print("Fail to find a valid grasp pose")

    #print(env.cube_half_size[2]*2, obb.primitive.extents[2]/2)

    #from pdb import set_trace
    #set_trace()
    return grasp_pose, obb

def solve(env: StackCubeEnv, seed=None, debug=False, vis=False, dry_run=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    grasp_pose, obb = get_grasp_pose_and_obb(env)
    env = env.unwrapped
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])    
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    height = env.cube_half_size[2] * 2
    height = obb.primitive.extents[2]/2 + get_actor_obb(env.cubeB).primitive.extents[2]/2 + 0.001
#    height = 0
    goal_pose = env.cubeB.pose * sapien.Pose([0, 0, height])
    offset = (goal_pose.p - env.cubeA.pose.p).numpy()[0] # remember that all data in ManiSkill is batched and a torch tensor
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)

    if dry_run:
        return [reach_pose, grasp_pose, "close_gripper", lift_pose, align_pose, "open_gripper"]

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    # -------------------------------------------------------------------------- #
    # Reach
    planner.move_to_pose_with_screw(reach_pose)
    # Grasp
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    # Lift
    planner.move_to_pose_with_screw(lift_pose)
    # Stack
    planner.move_to_pose_with_screw(align_pose)
    res = planner.open_gripper()
    planner.close()
    return res
