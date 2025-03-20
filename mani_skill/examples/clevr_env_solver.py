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

    if env.cubeB.name.startswith("clevr"):
        compute_grasp_info_by_obb_func = compute_grasp_info_by_obb_clevr
        print("YYY")
    else:
        compute_grasp_info_by_obb_func = compute_grasp_info_by_obb

    grasp_info = compute_grasp_info_by_obb_func(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    # this builds a grasp, which has position of center
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

from mani_skill.utils import common
import trimesh
def compute_grasp_info_by_obb_clevr(
    obb: trimesh.primitives.Box,
    approaching=(0, 0, -1),
    target_closing=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    # Assume normalized
    approaching = np.array(approaching)

    # Find the axis closest to approaching vector
    angles = approaching @ T[:3, :3]  # [3]
    inds0 = np.argsort(np.abs(angles))
    ind0 = inds0[-1]

    # Find the shorter axis as closing vector
    inds1 = np.argsort(extents[inds0[0:-1]])
    ind1 = inds0[0:-1][inds1[0]]
    ind2 = inds0[0:-1][inds1[1]]

    # If sizes are close, choose the one closest to the target closing
    if target_closing is not None and 0.99 < (extents[ind1] / extents[ind2]) < 1.01:
        vec1 = T[:3, ind1]
        vec2 = T[:3, ind2]
        if np.abs(target_closing @ vec1) < np.abs(target_closing @ vec2):
            ind1 = inds0[0:-1][inds1[1]]
            ind2 = inds0[0:-1][inds1[0]]
    closing = T[:3, ind1]

    # Flip if far from target
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    # Reorder extents
    extents = extents[[ind0, ind1, ind2]]

    # Find the origin on the surface
    center = T[:3, 3].copy()
    half_size = extents[0] * 0.5

    # This was hte old code, didn't work with long unsymetric shapes
    #center = center + approaching * (-half_size + min(depth, half_size))

    if depth < half_size: # if extents is long, subtract grasp depth from extent
        center = center + (-1)*approaching *(extents[0] - depth)
    else:  # if extents is short, just grasp in the middle
        center = center + (-1)*approaching *(half_size)

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = common.np_normalize_vector(closing)

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info





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
