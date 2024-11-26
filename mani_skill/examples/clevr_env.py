"""

"""
from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.io_utils import load_json

from pdb import set_trace

@register_env("ClevrMove-v1", max_episode_steps=50)
class StackCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    # Geometric sahpes, inspired by CLEVR
    shapes = {"sphere":actors.build_sphere, "cube":actors.build_cube } # "cylinder":actors.build_cylinder}
    colors = {"gray": [87, 87, 87],  "red": [173, 35, 35], "blue": [42, 75, 215], "green": [29, 105, 20], "brown": [129, 74, 25], "purple": [129, 38, 192], "cyan": [41, 208, 208], "yellow": [255, 238, 51]}
    sizes = {"large": 0.7/10./2., "small": 0.35/10./2.}

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        try:
            self.ycb_model_ids = np.array(
                list(
                    load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
                )
            )
        except:
            print('Warning YCB objects not found, try: python -m mani_skill.examples.demo_random_action -e "PickSingleYCB-v1" --render-mode="human"')
            self.ycb_model_ids = np.array(())
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # TODO: figure out a good way to calculate this
        start_p = [0.6, 0.7, 0.6]
        end_p = [0.0, 0.0, 0.12]
        t = 0.5
        new_p = (np.array(start_p)*t + np.array(end_p)*(1-t)).tolist()
        pose = sapien_utils.look_at(new_p, end_p)
        return CameraConfig("render_camera", pose, 448, 448, 1, 0.01, 100)

    def _load_scene_clevr(self, num_objects, min_unique=2, max_attempts=100):
        # Make sure that there are at least min_unique unique (non-duplicate) objects
        for _ in range(max_attempts):
            shapes_choice = randomization.uniform(0.0, float(len(self.shapes)),size=(num_objects,)).cpu().numpy().astype(int)
            colors_choice = randomization.uniform(0.0, float(len(self.colors)),size=(num_objects,)).cpu().numpy().astype(int)
            sizes_choice = randomization.uniform(0.0, float(len(self.sizes)),size=(num_objects,)).cpu().numpy().astype(int)
            shape_array = np.array((shapes_choice, colors_choice, sizes_choice)).T
            unique, counts = np.unique(shape_array, axis=0, return_counts=True)
            if np.sum(counts==1) >= min_unique:
                break
        if np.sum(counts==1) < min_unique:
            print(f"Failed to sample {min_unique} unique objects after {max_attempts} attempts.") 
        # NumPy arrays don't handle element-wise comparisons with the Python in operator directly when comparing arrays.
        # so do this as tuples
        unique_set = set(map(tuple, unique[np.where(counts==1)[0]]))
        self.objects_unique = [tuple(row) in unique_set for row in shape_array]

        self.objects = []
        self.objects_descr = []
        for i in range(num_objects):
            shape_name, build_function = list(self.shapes.items())[shapes_choice[i]]
            color_name, color = list(self.colors.items())[colors_choice[i]]
            size_name, size = list(self.sizes.items())[sizes_choice[i]]
            color = list(np.array(color + [255.,])/255.)
            tmp = build_function(self.scene, size, color=color, name=f"{shape_name}_{i}")
            self.objects.append(tmp)
            # now do text description
            descr = dict(shape=shape_name,
                         size=size_name, 
                         color=color_name)
            self.objects_descr.append(descr)
    
    def _load_scene_ycb(self, num_objects):
        model_ids = randomization.uniform(0.0, float(len(self.ycb_model_ids)),size=(num_objects,)).cpu().numpy().astype(int)
        model_ids = [2, 2]  # TODO(max): fix grasping, also 4
        for i, model_idx in enumerate(model_ids):
            model_id = self.ycb_model_ids[model_idx]
            #model_id = "043_phillips_screwdriver"
            # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            #builder.set_scene_idxs([i])
            model_name = " ".join(model_id.split("_")[1:])
            self.objects.append(builder.build(name=f"{model_id}-{i}"))
            self.objects_descr.append(dict(size="", color="", shape=model_name))

        # save if objects are unique
        unique_vals, counts = np.unique(model_ids, return_counts=True)
        count_dict = dict(zip(unique_vals, counts))
        self.objects_unique = [count_dict[num] == 1 for num in model_ids]

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        min_objects = 2
        max_objects = 5

        num_objects = int(randomization.uniform(float(min_objects), float(max_objects+1), size=(1,)))
        
        self.objects = []
        self.objects_descr = []
        #self._load_scene_ycb(num_objects)
        self._load_scene_clevr(num_objects)

        self.cubeA = self.objects[0]  
        self.cubeB = self.objects[1]
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]

            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            for shape, descr in zip(self.objects, self.objects_descr):
                shape_xy = xy + sampler.sample(radius, max_trials=100)
                xyz[:, :2] = shape_xy
                #xyz[:, 2] = self.sizes[descr["size"]]
                #xyz[:, 2] = 0
                qs = randomization.random_quaternions(
                    b,
                    lock_x=True,
                    lock_y=True,
                    lock_z=False,
                )
                shape.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))


    def set_goal_pose(self, objA_goal_pose):
        # Move cubeA onto cubeB
        self.objA_goal_pose = objA_goal_pose
        self.objA_to_goal_dist_inital = torch.linalg.norm(self.cubeA.pose.p - objA_goal_pose.p, axis=1)

    def eval_reward(self):
        objA_pose = self.cubeA.pose.p
        objA_to_goal_dist = torch.linalg.norm(objA_pose - self.objA_goal_pose.p, axis=1)
        reward = torch.clamp(1 - objA_to_goal_dist / self.objA_to_goal_dist_inital, 0, 1)
        return reward
        
    # This is the old code below. It is not used in the current implementation.
    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
