"""
See examples/motionplanning/panda/solutions/stack_cube.py for the template of this
"""
import gymnasium as gym
import numpy as np

from mani_skill.envs.tasks import StackCubeEnv
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


def object_is_rotationally_invariant(env, obj):
    if env.object_dataset == "clevr" and "sphere" in obj.name:
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

    # build a grasp, which has position of center
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # TODO(max): if object is z-rot invariant, keep current tcp orn
    # this is very simplified, but works for now
    if object_is_rotationally_invariant(env, env.cubeA):
        grasp_pose.set_q(env.agent.tcp.pose.get_q()[0])

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
    return grasp_pose, obb
