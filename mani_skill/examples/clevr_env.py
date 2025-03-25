"""

"""
from typing import Any, Dict, Union, Optional

import numpy as np
import torch
import sapien

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.sensors.depth_camera import StereoDepthCameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.ai2thor import ProcTHORSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.io_utils import load_json
from mani_skill.examples.utils_env_interventions import move_object_onto
from mani_skill.examples.utils_traj_tokens import to_prefix_suffix
from mani_skill.examples.utils_traj_tokens import getActionEncDecFunction
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb


from pdb import set_trace

@register_env("ClevrMove-v1", max_episode_steps=50)
class StackCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_OBJECT_DATASETS = ["clevr", "ycb", "objaverse"]
    SUPPORTED_SCENE_DATASETS = ['Table', 'ProcTHOR']
    agent: Union[Panda, Fetch]



    def __init__(
        self, *args, robot_uids="panda_wristcam", scene_dataset="Table", object_dataset="clevr", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.scene_dataset = scene_dataset
        self.object_dataset = object_dataset
        # TODO [NH] This seems not as intended? I would expect it was intended to go 
        # from -0.1 to 0.1 and -0.2 to 0.2. But now it goes from -0.1 to -0.2 and 0.1 to 0.2
        # self.object_region = [[-0.1, -0.2], [0.1, 0.2], [.12,.12]]
        # TODO This is the fix:
        if object_dataset in ["clevr", "ycb"]:
            self.object_region = np.array([[-0.1, 0.1], [-0.2, 0.2], [.12,.12]])
        elif object_dataset == "objaverse":
            # Larger objects --> more space to sample from
            self.object_region = np.array([[-0.3, 0.1], [-0.2, 0.2], [.12,.12]])
        #self.cam_size = 224
        self.cam_size = 448

        self.RANDOMIZE_VIEW = True
        self.RESAMPLE_CAMERA_IF_OBJS_UNSEEN = 100

        # cached stuff for loaders
        self.objects = None
        self.objaverse_model_ids = None
        self.ycb_model_ids = None
        self.spok_dataset = None

        self.initalize_render_camera_fixed()  # sets render_camera_config
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def base_camera_pose(self):
        #pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        start_p = [0.6, 0.7, 0.6]
        end_p = [0.0, 0.0, 0.12]
        t = 0.5
        new_p = (np.array(start_p)*t + np.array(end_p)*(1-t)).tolist()
        pose = sapien_utils.look_at(new_p, end_p)
        return pose
        
    def initalize_render_camera_fixed(self):
        start_p = [0.6, 0.7, 0.6]
        end_p = [0.0, 0.0, 0.12]
        t = 0.5
        new_p = (np.array(start_p)*t + np.array(end_p)*(1-t)).tolist()
        pose = sapien_utils.look_at(new_p, end_p)
        self.render_camera_config = CameraConfig("render_camera", pose,  self.cam_size,  self.cam_size, 1, 0.01, 100)
        #self.render_camera_config = StereoDepthCameraConfig("render_camera", pose,  self.cam_size,  self.cam_size, 1, 0.01, 100)

    def initalize_render_camera_random(self):
        cylinder_c = np.array([.45, 0, .36])
        cylinder_ext = np.array([.10, np.pi*4/5, .10])
        cylinder_l = cylinder_c - cylinder_ext
        cylinder_h = cylinder_c + cylinder_ext
        r, phi, z = randomization.uniform(cylinder_l, cylinder_h, size=(3,)).cpu().numpy().astype(float)
        start_p = [r * np.cos(phi), r * np.sin(phi), z]
        end_p = randomization.uniform(*zip(*self.object_region),size=(3,)).cpu().numpy().astype(float)
        
        # TODO(max): fix this
        if self.scene_dataset == "ProcTHOR":
            start_p = (np.array(start_p) + np.array(end_p)).tolist()

        pose = sapien_utils.look_at(start_p, end_p)
        self.render_camera_config = CameraConfig("render_camera", pose,  self.cam_size,  self.cam_size, 1, 0.01, 100)
        #self.render_camera_config = StereoDepthCameraConfig("render_camera", pose,  self.cam_size,  self.cam_size, 1, 0.01, 100)

    @property
    def _default_sensor_configs(self):
        return self.render_camera_config

    @property
    def _default_human_render_camera_configs(self):
        return self.render_camera_config
        
    def _load_agent(self, options: dict, initial_agent_poses: Optional[Union[sapien.Pose, Pose]] = sapien.Pose(p=[0.0, 0, 0])):
        super()._load_agent(options, initial_agent_poses) 

    def _load_scene_clevr(self, num_objects, min_unique=2, max_attempts=100):
        # Geometric sahpes, inspired by CLEVR
        shapes = {"sphere":actors.build_sphere, "cube":actors.build_cube, "box":actors.build_box} # "cylinder":actors.build_cylinder}
        colors = {"gray": [87, 87, 87],  "red": [173, 35, 35], "blue": [42, 75, 215], "green": [29, 105, 20],
                  "brown": [129, 74, 25], "purple": [129, 38, 192], "cyan": [41, 208, 208], "yellow": [255, 238, 51]}
        sizes = {"large": 0.7/10./2., "small": 0.35/10./2.}
        # Make sure that there are at least min_unique unique (non-duplicate) objects
        assert max_attempts > 0
        unique, counts = None, None
        for _ in range(max_attempts):
            shapes_choice = randomization.uniform(0.0, float(len(shapes)),size=(num_objects,)).cpu().numpy().astype(int)
            colors_choice = randomization.uniform(0.0, float(len(colors)),size=(num_objects,)).cpu().numpy().astype(int)
            sizes_choice = randomization.uniform(0.0, float(len(sizes)),size=(num_objects,)).cpu().numpy().astype(int)
            upright_choice = randomization.uniform(0.0, float(2),size=(num_objects,)).cpu().numpy().astype(int)
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
            shape_name, build_function = list(shapes.items())[shapes_choice[i]]
            color_name, color = list(colors.items())[colors_choice[i]]
            size_name, size = list(sizes.items())[sizes_choice[i]]
            color = list(np.array(color + [255.,])/255.)
            initial_pose = sapien.Pose(p=[0, 0, 0.02], q=[1, 0, 0, 0])
            name = f"clevr_{shape_name}_{i}"
            if shape_name == "box":
                if upright_choice[i] == 0:
                    half_extents = (2*size, size, size)
                else:
                    half_extents = (size, size, 2*size)
                tmp = build_function(self.scene, half_extents, color=color, name=name, initial_pose=initial_pose)
            else:
                tmp = build_function(self.scene, size, color=color, name=name, initial_pose=initial_pose)
            self.objects.append(tmp)
            # now do text description
            descr = dict(shape=shape_name,
                         size=size_name, 
                         color=color_name)
            self.objects_descr.append(descr)
    
    def _load_scene_ycb(self, num_objects):
        if self.ycb_model_ids is None:
            ycb_file = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
            try:
                self.ycb_model_ids = np.array(list(load_json(ycb_file).keys()))
            except:
                raise ValueError('Warning YCB objects not found, try: python -m mani_skill.examples.demo_random_action -e "PickSingleYCB-v1" --render-mode="human"')
                
        model_ids = randomization.uniform(0.0, float(len(self.ycb_model_ids)),size=(num_objects,)).cpu().numpy().astype(int)
        #model_ids = [2, 2]  # TODO(max): fix grasping, also 4
        for i, model_idx in enumerate(model_ids):
            model_id = self.ycb_model_ids[model_idx]
            #model_id = "043_phillips_screwdriver"
            # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.02], q=[1, 0, 0, 0])
            #builder.set_scene_idxs([i])
            model_name = " ".join(str(model_id).split("_")[1:])
            self.objects.append(builder.build(name=f"{model_id}-{i}"))
            self.objects_descr.append(dict(size="", color="", shape=model_name))

        # save if objects are unique
        unique_vals, counts = np.unique(model_ids, return_counts=True)
        count_dict = dict(zip(unique_vals, counts))
        self.objects_unique = [count_dict[num] == 1 for num in model_ids]

    def _load_scene_objaverse(self, num_objects: int=2):
        # TODO Refactor to make this explit as Spok?
        from mani_skill.examples.objaverse_handler import SpokDatasetBuilderFast,SpokDatasetBuilder, get_spok_builder
        if self.spok_dataset is None:
            #self.spok_dataset = SpokDatasetBuilder(maximum_objects=20_000)
            self.spok_dataset = SpokDatasetBuilderFast(maximum_objects=20_000)

        uuids = self.spok_dataset.sample_uuids(num_objects, with_replacement=False)
        #uuids = ['3239828896624cdaae337e1b8b5ca78f', '2bcfd1118fd245ef88c838f46960d4b3', '54a13e432b9a488ab35d1c6644d9bc0c']

        for uuid in uuids:
            obj_builder = get_spok_builder(self.scene, uuid, add_collision=True, add_visual=True,
                                           spok_dataset=self.spok_dataset)
            model_name = f"{uuid}"
            shape_name = f"{uuid}"
            try:
                shape_name = self.spok_dataset.get_object_name(uuid) # default is three_words
            except Exception as e:
                print(f"Could not find CLIP name for {uuid}, using category attribute as {shape_name}")
                print(f"Exception {e =}")
                try:
                    shape_name = self.spok_dataset.get_gpt_name(uuid) # default is three_words
                except Exception as e:
                    print(f"Could not find GPT name for {uuid}, using category attribute as {shape_name}")
                    print(f"Exception {e =}")

            self.objects.append(obj_builder.build(name=f"{model_name}"))
            self.objects_descr.append(dict(size="", color="", shape=shape_name))

            # from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
            # print("sizes", model_name, get_actor_obb(self.objects[-1]).primitive.extents)
        
        #set_trace()

        unique_vals, counts = np.unique(uuids, return_counts=True)
        count_dict = dict(zip(unique_vals, counts))
        self.objects_unique = [count_dict[num] == 1 for num in uuids]

    def _load_scene(self, options: dict):
        if self.scene_dataset == "Table":
            self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            self.table_scene.build()
        elif self.scene_dataset == "ProcTHOR":
            self.table_scene = ProcTHORSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            self.table_scene.build(build_config_idxs=0)

            ref_item = self.table_scene.scene_objects["env-0_objects/Toaster_6_2"]
            ref_pose = dict(self.table_scene._default_scene_objects_poses)[ref_item]
            region_pos = ref_pose.get_p()
            print("ref pose", region_pos)
            extents = [.1, .1, 0]
            region = [region_pos + extents, region_pos - extents]
            self.object_region = np.array(region).T.tolist()
            ref_item.remove_from_scene()

            # add lighting
            ray_traced_lighting = self._custom_human_render_camera_configs.get("shader_pack",None) in ["rt","rt-fast"]
            self.scene.set_ambient_light([3 if ray_traced_lighting else 0.3] * 3)
            color = np.array([1.0, 0.8, 0.5]) * (10 if ray_traced_lighting else 2)
            self.scene.add_point_light([region_pos[0], region_pos[1], 2.3], color=color)
        else:
            raise ValueError

        min_objects = 2
        max_objects = 5
        num_objects = int(randomization.uniform(float(min_objects), float(max_objects+1), size=(1,)))
        
        # Turn off gravity
        self.scene.sim_config.scene_config.gravity = np.array([0,0,0])
        
        self.objects = []
        self.objects_descr = []
        if self.object_dataset == "clevr":
            self._load_scene_clevr(num_objects)
        elif self.object_dataset == "ycb":
            self._load_scene_ycb(num_objects)
        elif self.object_dataset == "objaverse":
            self._load_scene_objaverse(num_objects)
        else:
            raise ValueError
        
        assert len(self.objects) == num_objects, f"Expected {num_objects} objects, got {len(self.objects)}"
        assert len(self.objects_descr) == num_objects, f"Expected {num_objects} objects, got {len(self.objects_descr)}"
        
        self.cubeA = self.objects[0]  
        self.cubeB = self.objects[1]
        
    def check_objects_visible(self, obj_a, obj_b, camera):
        """
        Returns True if objects are visible from camera, else return False.
        """
        # TODO(max): This should really be fixed at some point.
        # just make use of the encoding function for now, at some point write a function
        grasp_pose = obj_a
        tcp_pose = obj_a
        action_text = "check visibility"
        enc_func, dec_func = getActionEncDecFunction("xyzrotvec-cam-proj2")
        prefix, token_str, curve_3d, orns_3d, info = to_prefix_suffix(obj_a, obj_b,
                                                                camera, grasp_pose, tcp_pose,
                                                                action_text, enc_func, robot_pose=None)
        if info["didclip_traj"]:
            return False
        return True
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            if self.scene_dataset == "ProcTHOR":
                xyz[:, 2] = self.object_region[2][0]
            else:
                xyz[:, 2] = 0.0 #0.02
                # Random global shift
                xy = torch.rand((b, 2)) * 0.2 - 0.1 # rand is uniform [0,1], so shift to [-0.1, 0.1]

            # We have to flip the object region as it is defined as 
            #   [[min_1, max_1], [min_2, max_2], ...]
            # but the sampler expects
            #   [[min_1, min_2, ...], [max_1, max_2, ...]]
            sampler = randomization.UniformPlacementSampler(bounds=self.object_region[:2].T, batch_size=b)
            
            # TODO [NH] This is not good, radius should depend on the object 
            # radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001 # Radius is get
            
            for shape, descr in zip(self.objects, self.objects_descr):
                # Get radius of object
                radius = np.linalg.norm(get_actor_obb(shape).primitive.extents) / 2
                if self.scene_dataset == "ProcTHOR":
                    region_mins = [x[0] for x in self.object_region]
                    region_maxs = [x[1] for x in self.object_region]
                    xyz[:,] = randomization.uniform(region_mins, region_maxs, size=(1,))
                    xyz[:,2] += 0.02
                else:
                    shape_xy = xy + sampler.sample(radius, max_trials=100, verbose=False)
                    xyz[:, :2] = shape_xy

                qs = randomization.random_quaternions(
                    b,
                    lock_x=True,
                    lock_y=True,
                    lock_z=False,
                )
                shape.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            
            # do intervention
            obj_start, obj_end, action_text = move_object_onto(self, pretend=True)
            self.set_goal_pose(obj_end)
            self.obj_start = obj_start
            self.obj_end = obj_end
            self.action_text = action_text
            from mani_skill.examples.clevr_env_solver import get_grasp_pose_and_obb    
            grasp_pose, _ = get_grasp_pose_and_obb(self)
            self.grasp_pose = Pose.create_from_pq(p=grasp_pose.get_p(), q=grasp_pose.get_q())

            are_visible = False
            if self.RANDOMIZE_VIEW:
                self.initalize_render_camera_random()
                for i in range(self.RESAMPLE_CAMERA_IF_OBJS_UNSEEN):
                    camera = self.scene.human_render_cameras['render_camera'].camera 
                    #print("XXX", camera.get_extrinsic_matrix())
                    are_visible = self.check_objects_visible(obj_start, obj_end, camera)
                    if are_visible:
                        break
                    self.initalize_render_camera_random()
                    self._setup_sensors(options)
                if not are_visible:
                    print("Warning: could not sample visible camera position.")
                
            else:
                self.initalize_render_camera_fixed()


    def set_goal_pose(self, objA_goal_pose):
        # Move cubeA onto cubeB
        self.objA_goal_pose = objA_goal_pose
        self.objA_to_goal_dist_inital = torch.linalg.norm(self.cubeA.pose.p - objA_goal_pose.p, axis=1)

    # For the old rewards see: ../mani_skill/envs/tasks/tabletop/stack_cube.py
    def eval_reward(self):
        objA_pose = self.cubeA.pose.p
        objA_to_goal_dist = torch.linalg.norm(objA_pose - self.objA_goal_pose.p, axis=1)
        reward = torch.clamp(1 - objA_to_goal_dist / self.objA_to_goal_dist_inital, 0, 1)
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.eval_reward()

    def _get_obs_extra(self, info: Dict):
        tcp_pose = self.agent.tcp.pose
        robot_pose = self.agent.robot.get_root_pose()
        obs = dict(obj_start=self.obj_start.raw_pose, obj_end=self.obj_end.raw_pose,
                   grasp_pose=self.grasp_pose.raw_pose, tcp_pose=tcp_pose.raw_pose, robot_pose=robot_pose.raw_pose)
        obs["action_text_"+self.action_text] = 123
        return obs
    
    def reset(self, seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None):
        if self.object_dataset == "clevr":
            pass
        elif self.object_dataset == "ycb" or self.object_dataset == "objaverse":
            if self.objects is not None:
                for object in self.objects:
                    object.remove_from_scene()
                    del object
        else:
            raise ValueError
        #set_trace()
        return super().reset(seed=seed, options=options)
