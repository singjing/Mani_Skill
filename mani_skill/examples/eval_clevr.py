
import gymnasium as gym
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Pose
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

import clevr_env  # do import to register env, not used otherwise
from utils_env_interventions import move_object_onto
from utils_trajectory import project_points, generate_curve_torch, plot_gradient_curve

# [NH] This is missing?
# from gen_dataset import Args, reset_random
from run_env import Args, reset_random

from pdb import set_trace

class EnvWrapper:
    def __init__(self, parallel_in_single_scene=False):

        args = Args()
        args.env_id = "ClevrMove-v1"
        args.control_mode = "pd_joint_pos"
        args.object_dataset = "objaverse"
        #args.num_envs = 2
        self.env = gym.make(
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
            object_dataset=args.object_dataset,
            parallel_in_single_scene=parallel_in_single_scene,
            # **args.env_kwargs
        )
        reset_random(args, orig_seeds=None)
        self.args = args
        self.env_idx = 0
    

    def reset(self):
        obs, _ = self.env.reset(seed=self.args.seed[0], options=dict(reconfigure=True))
        self.env.render()
        images = self.env.base_env.scene.get_human_render_camera_images('render_camera')
        image_before = images['render_camera'][self.env_idx].detach().cpu().numpy()

        # do intervention ( this will set env.base_env.cubeA / cubeB)
        obj_start_pose, obj_end_pose, action_text = move_object_onto(self.env, pretend=True)
        self.env.unwrapped.set_goal_pose(obj_end_pose)

        # convert to default trajectory
        self.camera = self.env.base_env.scene.human_render_cameras['render_camera'].camera

        return (image_before, action_text)

    def step(self, action, vis=True):
        curve_3d, orns_3d = action
        assert curve_3d.shape[1] == 2 and orns_3d.shape[1] == 2  # start and stop poses
        _, curve_3d_i = generate_curve_torch(curve_3d[:, 0], curve_3d[:, -1], num_points=3)
        grasp_pose = Pose.create_from_pq(p=curve_3d[:, 0], q=orns_3d[:, 0])
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        lift_pose = Pose.create_from_pq(p=curve_3d_i[:, 1], q=orns_3d[:, 1])
        align_pose = Pose.create_from_pq(p=curve_3d_i[:, 2], q=orns_3d[:, 1])

        planner = PandaArmMotionPlanningSolver(
            self.env,
            debug=False,
            vis=vis,
            base_pose=self.env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=vis,
            print_env_info=False,
        )
        planner.move_to_pose_with_screw(reach_pose)
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        planner.move_to_pose_with_screw(lift_pose)
        planner.move_to_pose_with_screw(align_pose)
        res = planner.open_gripper()
        print("reward", self.env.unwrapped.eval_reward())
        planner.close()
        return res


if __name__ == "__main__":
    import torch
    env = EnvWrapper()
    image_before = env.reset()
    curve_3d = torch.tensor([[[-0.0232,  0.0247,  0.0175],
                              [-0.1308,  0.0793,  0.0390]]])
    orns_3d = torch.tensor([[[0.0000, 0.5323, 0.8465, 0.0000],
                             [0.0000, 0.5323, 0.8465, 0.0000]]])
    action = (curve_3d, orns_3d)
    result = env.step(action)
    print(result)
    print("done")
