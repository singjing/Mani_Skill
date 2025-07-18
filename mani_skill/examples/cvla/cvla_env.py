"""

"""
from typing import Any, Dict, Union, Optional

import numpy as np
import torch
import sapien
from scipy.spatial.transform import Rotation as R

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
# from mani_skill.sensors.depth_camera import StereoDepthCameraConfig
from mani_skill.utils import sapien_utils  # common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.ai2thor import ProcTHORSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.io_utils import load_json
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.examples.cvla.utils_env_interventions import move_object_onto, get_actor_mesh
from mani_skill.examples.cvla.cvla_env_solver import get_grasp_pose_and_obb
from mani_skill.examples.cvla.utils_traj_tokens import to_prefix_suffix, getActionEncInstance
from mani_skill.examples.cvla.objaverse_handler import SpocDatasetBuilderFast, get_spoc_builder

obj_start_globle = [0, 0, 0]


@register_env("CvlaMove-v1", max_episode_steps=50)
class CvlaMoveEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_OBJECT_DATASETS = ["clevr", "ycb", "objaverse"]
    SUPPORTED_SCENE_DATASETS = ['Table', 'ProcTHOR']
    SUPPORTED_CAM_VIEWS = ['fixed', 'random_side', 'random', 'top']
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", scene_dataset="Table", object_dataset="clevr",
        camera_views="random", scene_options="fixed", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.scene_dataset = scene_dataset
        self.object_dataset = object_dataset
        if object_dataset in ["clevr", "ycb"]:
            self.object_region = np.array([[-0.1, 0.1], [-0.2, 0.2], [.12, .12]])
        elif object_dataset == "objaverse":
            # Larger objects --> more space to sample from
            self.object_region = np.array([[-0.3, 0.1], [-0.2, 0.2], [.12, .12]])
        self.cam_size = 448  # or 224

        self.camera_views = camera_views
        self.cam_resample_if_objs_unseen = 100  # unseen as in not in cam fustrum
        self.scene_options = scene_options

        # cached stuff for loaders
        self.objects = []
        self.objaverse_model_ids = None
        self.ycb_model_ids = None
        self.spoc_dataset = None

        self.initalize_render_camera()  # sets render_camera_config
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def base_camera_pose(self):
        # pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        start_p = [0.6, 0.7, 0.6]
        end_p = [0.0, 0.0, 0.12]
        t = 0.5
        new_p = (np.array(start_p) * t + np.array(end_p) * (1 - t)).tolist()
        pose = sapien_utils.look_at(new_p, end_p)
        return pose

    def initalize_render_camera(self):
        fov_range = [1.0, 1.0]
        z_range = [0.0, 0.0]
        grasp_pose = [0, 0, 0]
        if self.camera_views == "fixed":
            start_p = [0.6, 0.7, 0.6]
            end_p = [0.0, 0.0, 0.12]
            t = 0.5
            start_p = (np.array(start_p) * t + np.array(end_p) * (1 - t)).tolist()
        
        elif self.camera_views == "top" :
            
            if hasattr(self, 'grasp_pose'):
                # Define cylindrical sampling parameters for top view
                obj_start, obj_end, action_text = move_object_onto(self, pretend=True)
                #print(action_text)
                grasp_pose = obj_start.p
                grasp_pose = grasp_pose[0]
                # Convert to Cartesian coordinates for camera position
                start_p = [
                    grasp_pose[0], #+ r * np.cos(phi), # so the virtual camera will look from the robot arm and avoid occlusion
                    grasp_pose[1], #+ r * np.sin(phi),
                    grasp_pose[2] + 1
                ]
                # Camera looks downward (same XY as start_p but lower Z)
                end_p = [start_p[0], start_p[1], start_p[2] - 0.3]
                
            else: # the first time before grasp_pose was generate, make the camera at the zero point
                start_p = [0.0, 0.0, 0.6]
                end_p = [0.0, 0.0, 0.0]
            
        else:
            #tmp test
            #self.camera_views = "random_side"
            if self.camera_views == "random_side":
                cylinder_l = np.array([.35, -np.pi * 4 / 5, .26])
                cylinder_h = np.array([.55, np.pi * 4 / 5, .46])
                if hasattr(self, 'grasp_pose'):
                    obj_start, obj_end, action_text = move_object_onto(self, pretend=True)
                    grasp_pose = obj_start.p
                    grasp_pose = grasp_pose[0]
            elif self.camera_views == "random":
                cylinder_l = np.array([.0, -np.pi, .25])
                cylinder_h = np.array([.50, np.pi, .65])
                            
            elif self.camera_views == "random_fov":
                cylinder_l = np.array([.0, -np.pi, .25])
                cylinder_h = np.array([.50, np.pi, .65])
                fov_range = [np.deg2rad(50), np.deg2rad(75)]
                z_range = [-15, 15]
            else:
                #raise ValueError(f"unknown camera_views {self.camera_views}, options {self.SUPPORTED_CAM_VIEWS}")
                #only for grasp task
                cylinder_l = np.array([0.1, -np.pi, 0.4])  # min radius, min angle, min height
                cylinder_h = np.array([0.3, np.pi, 0.6])   # max radius, max angle, max height
                
            r, phi, z = randomization.uniform(cylinder_l, cylinder_h, size=(3,)).cpu().numpy().astype(float)
            start_p = [r * np.cos(phi), r * np.sin(phi), z]
            end_p = randomization.uniform(*zip(*self.object_region), size=(3,)).cpu().numpy().astype(float)
        
        fov = randomization.uniform(*fov_range, size=(1,)).cpu().numpy().astype(float)[0]
        z_rot = randomization.uniform(*z_range, size=(1,)).cpu().numpy().astype(float)[0]
        z_rot_orn = R.from_euler("xyz", (0, 0, z_rot), degrees=True)
        z_rot_pose = Pose.create_from_pq(q=z_rot_orn.as_quat(scalar_first=True))

        if self.scene_dataset == "ProcTHOR":
            print("Warning: ProcTHOR camera randomization not well tested.")
            start_p = (np.array(start_p) + np.array(end_p)).tolist()

        pose = sapien_utils.look_at(start_p, end_p) * z_rot_pose
        print("pose")
        print(pose)
        self.render_camera_config = CameraConfig("render_camera", pose, width=self.cam_size, height=self.cam_size,
                                                 fov=fov, near=0.01, far=100)
        # self.render_camera_config = StereoDepthCameraConfig("render_camera", pose,  self.cam_size,  self.cam_size, 1, 0.01, 100)

    @property
    def _default_sensor_configs(self):
        return self.render_camera_config

    @property
    def _default_human_render_camera_configs(self):
        return self.render_camera_config

    def _load_agent(self, options: dict, initial_agent_poses: Optional[Union[sapien.Pose, Pose]] = sapien.Pose(p=[0.0, 0, 0]), build_separate: bool = False):
        initial_agent_poses = sapien.Pose(p=[0.0, 0, 300])
        super()._load_agent(options, initial_agent_poses, build_separate)

    def _load_scene_clevr(self, num_objects, min_unique=2, max_attempts=100):
        # Geometric sahpes, inspired by CLEVR
        shapes = {"sphere": actors.build_sphere, "cube": actors.build_cube, "box": actors.build_box}  # "cylinder":actors.build_cylinder}
        colors = {"gray": [87, 87, 87], "red": [173, 35, 35], "blue": [42, 75, 215], "green": [29, 105, 20],
                  "brown": [129, 74, 25], "purple": [129, 38, 192], "cyan": [41, 208, 208], "yellow": [255, 238, 51]}
        sizes = {"large": 0.7 / 10. / 2., "small": 0.35 / 10. / 2.}
        # Make sure that there are at least min_unique unique (non-duplicate) objects
        assert max_attempts > 0
        unique, counts = np.empty((0,), dtype=int), np.empty((0,), dtype=int)  # for pylance

        for _ in range(max_attempts):
            shapes_choice = randomization.uniform(0.0, float(len(shapes)), size=(num_objects,)).cpu().numpy().astype(int)
            colors_choice = randomization.uniform(0.0, float(len(colors)), size=(num_objects,)).cpu().numpy().astype(int)
            sizes_choice = randomization.uniform(0.0, float(len(sizes)), size=(num_objects,)).cpu().numpy().astype(int)
            upright_choice = randomization.uniform(0.0, float(2), size=(num_objects,)).cpu().numpy().astype(int)
            shape_array = np.array((shapes_choice, colors_choice, sizes_choice)).T
            unique, counts = np.unique(shape_array, axis=0, return_counts=True)
            if np.sum(counts == 1) >= min_unique:
                break

        if np.sum(counts == 1) < min_unique:
            print(f"Failed to sample {min_unique} unique objects after {max_attempts} attempts.")
        # NumPy arrays don't handle element-wise comparisons with the Python in operator directly when comparing arrays.
        # so do this as tuples
        unique_set = set(map(tuple, unique[np.where(counts == 1)[0]]))
        self.objects_unique = [tuple(row) in unique_set for row in shape_array]
        assert num_objects >= 1
        self.objects = []
        self.object_names = []
        for i in range(num_objects):
            shape_name, build_function = list(shapes.items())[shapes_choice[i]]
            color_name, color = list(colors.items())[colors_choice[i]]
            size_name, size = list(sizes.items())[sizes_choice[i]]
            color = list(np.array(color + [255.,]) / 255.)
            initial_pose = sapien.Pose(p=[0, 0, 0.02], q=[1, 0, 0, 0])
            object_name = f'{size_name}_{color_name}_{shape_name}_{i}'.strip()
            if shape_name == "box":
                if upright_choice[i] == 0:
                    half_extents = (2 * size, size, size)
                else:
                    half_extents = (size, size, 2 * size)
                tmp = build_function(self.scene, half_extents, color=color, name=object_name, initial_pose=initial_pose)
            else:
                tmp = build_function(self.scene, size, color=color, name=object_name, initial_pose=initial_pose)
            self.objects.append(tmp)

            # now do text description
            object_name_nice = f'{size_name} {color_name} {shape_name}'
            self.object_names.append(object_name_nice)

    def _load_scene_ycb(self, num_objects):
        if self.ycb_model_ids is None:
            ycb_file = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
            try:
                self.ycb_model_ids = np.array(list(load_json(ycb_file).keys()))
            except FileNotFoundError:
                raise FileNotFoundError('Warning YCB objects not found, try: python -m mani_skill.examples.demo_random_action -e "PickSingleYCB-v1" --render-mode="human"')

        model_ids = randomization.uniform(0.0, float(len(self.ycb_model_ids)), size=(num_objects,)).cpu().numpy().astype(int)
        for i, model_idx in enumerate(model_ids):
            model_id = self.ycb_model_ids[model_idx]
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.02], q=[1, 0, 0, 0])
            # builder.set_scene_idxs([i])
            model_name = " ".join(str(model_id).split("_")[1:])
            self.objects.append(builder.build(name=f"{model_id}-{i}"))
            self.object_names.append(model_name)

        # save if objects are unique
        unique_vals, counts = np.unique(model_ids, return_counts=True)
        count_dict = dict(zip(unique_vals, counts))
        self.objects_unique = [count_dict[num] == 1 for num in model_ids]

    def _load_scene_objaverse(self, num_objects: int = 2):
        if self.spoc_dataset is None:
            self.spoc_dataset = SpocDatasetBuilderFast(maximum_objects=20_000)

        uuids = self.spoc_dataset.sample_uuids(num_objects, with_replacement=False)
        # uuids = ['3239828896624cdaae337e1b8b5ca78f', '2bcfd1118fd245ef88c838f46960d4b3', '54a13e432b9a488ab35d1c6644d9bc0c']

        for uuid in uuids:
            obj_builder = get_spoc_builder(self.scene, uuid, add_collision=True, add_visual=True,
                                           spoc_dataset=self.spoc_dataset)
            model_name = f"{uuid}"
            shape_name = f"{uuid}"
            try:
                shape_name = self.spoc_dataset.get_object_name(uuid)  # default is three_words
            except Exception as e:
                print(f"Could not find CLIP name for {uuid}, using category attribute as {shape_name}")
                print(f"Exception {e}")
                try:
                    shape_name = self.spoc_dataset.get_gpt_name(uuid)  # default is three_words
                except Exception as e:
                    print(f"Could not find GPT name for {uuid}, using category attribute as {shape_name}")
                    print(f"Exception {e}")

            self.objects.append(obj_builder.build(name=f"{model_name}"))
            self.object_names.append(shape_name)

        unique_vals, counts = np.unique(uuids, return_counts=True)
        count_dict = dict(zip(unique_vals, counts))
        self.objects_unique = [count_dict[num] == 1 for num in uuids]

    def _load_scene(self, options: dict):
        if self.scene_dataset == "Table":
            self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            if self.scene_options == "fixed":
                scene_choice = None
            else:
                num_options = self.table_scene.NUM_SCENE_OPTIONS
                scene_choice = [int(randomization.uniform(float(0), float(num_options), size=(1,)))]

            self.table_scene.build(scene_choice)

        elif self.scene_dataset == "ProcTHOR":
            self.table_scene = ProcTHORSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            self.table_scene.build(build_config_idxs=0)
            ref_item = self.table_scene.scene_objects["env-0_objects/Toaster_6_2"]
            ref_pose = dict(self.table_scene._default_scene_objects_poses)[ref_item]
            region_pos = ref_pose.get_p()
            print("ref pose", region_pos)
            extents = [.1, .1, 0]
            region = [region_pos + extents, region_pos - extents]
            self.object_region = np.array(region).T
            ref_item.remove_from_scene()

            # add lighting
            ray_traced_lighting = self._custom_human_render_camera_configs.get("shader_pack", None) in ["rt", "rt-fast"]
            self.scene.set_ambient_light([3 if ray_traced_lighting else 0.3] * 3)
            color = np.array([1.0, 0.8, 0.5]) * (10 if ray_traced_lighting else 2)
            self.scene.add_point_light([region_pos[0], region_pos[1], 2.3], color=color)
        else:
            raise ValueError

        min_objects = 2
        max_objects = 5
        num_objects = int(randomization.uniform(float(min_objects), float(max_objects + 1), size=(1,)))

        # Turn off gravity
        self.scene.sim_config.scene_config.gravity = np.array([0, 0, 0])

        self.objects = []
        self.object_names = []
        if self.object_dataset == "clevr":
            self._load_scene_clevr(num_objects)
        elif self.object_dataset == "ycb":
            self._load_scene_ycb(num_objects)
        elif self.object_dataset == "objaverse":
            self._load_scene_objaverse(num_objects)
        else:
            raise ValueError

        assert len(self.objects) == num_objects, f"Expected {num_objects} objects, got {len(self.objects)}"
        assert len(self.object_names) == num_objects, f"Expected {num_objects} objects, got {len(self.object_names)}"

        # set by intervention
        self.cubeA = None
        self.cubeB = None

    def check_objects_visible(self, obj_a, obj_b, camera):
        """
        Returns True if objects are visible from camera, else return False.
        """
        # TODO(max): This should really be fixed at some point.
        # just make use of the encoding function for now, at some point write a function
        grasp_pose = obj_a
        tcp_pose = obj_a
        action_text = "check visibility"
        action_encoder = getActionEncInstance("xyzrotvec-cam-1024xy")
        enc_func, _ = action_encoder.encode_trajectory, action_encoder.decode_trajectory
        _, _, _, _, info = to_prefix_suffix(obj_a, obj_b,
                                            camera, grasp_pose, tcp_pose,
                                            action_text, enc_func, robot_pose=None)
        if info["didclip_traj"]:
            return False
        return True

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        print("!!!")
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            assert isinstance(self.object_region, np.ndarray)
            # We have to flip the object region as it is defined as
            #   [[min_1, max_1], [min_2, max_2], ...]
            # but the sampler expects
            #   [[min_1, min_2, ...], [max_1, max_2, ...]]
            sampler = randomization.UniformPlacementSampler(bounds=self.object_region[:2].T, batch_size=b)

            for shape in self.objects:
                xyz = torch.zeros((b, 3))
                # Get radius of object
                radius = np.linalg.norm(get_actor_obb(shape).primitive.extents) / 2
                if self.scene_dataset == "ProcTHOR":
                    xyz[:, 2] = self.object_region[2][0]
                    region_mins = [x[0] for x in self.object_region]
                    region_maxs = [x[1] for x in self.object_region]
                    xyz[:,] = randomization.uniform(region_mins, region_maxs, size=(1,))
                    xyz[:, 2] += 0.02
                else:
                    # Random global shift
                    xy = torch.rand((b, 2)) * 0.2 - 0.1  # rand is uniform [0,1], so shift to [-0.1, 0.1]
                    shape_xy = xy + sampler.sample(radius, max_trials=10000, verbose=False, err_on_fail=True)
                    xyz[:, :2] = shape_xy

                    table = self.scene.actors['table-workspace']
                    table_z = get_actor_mesh(table, to_world_frame=True).vertices[:, 2].max()
                    shape_mesh = get_actor_mesh(shape, to_world_frame=True)
                    if self.object_dataset == "clevr":
                        height = (shape_mesh.vertices[:, 2].max() - shape_mesh.vertices[:, 2].min()) / 2
                    else:
                        height = 0  # load meshes, origin is at object bottom
                    xyz[:, 2] = table_z + height

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
            try:
                grasp_pose, _ = get_grasp_pose_and_obb(self)
            except AttributeError:
                print("Warning: finding grasp pose failed, using object pose")
                grasp_pose = obj_start
            try:
                self.agent.tcp
            except AttributeError:
                print("Warning: Robot has not TPC, using object pose")
                self.agent.tcp = type("Fake-TCP", (), {"pose": grasp_pose})()

            self.grasp_pose = Pose.create_from_pq(p=grasp_pose.get_p(), q=grasp_pose.get_q())

            are_visible = False
            self.initalize_render_camera()
            if self.camera_views != "fixed":
                for i in range(self.cam_resample_if_objs_unseen):
                    camera = self.scene.human_render_cameras['render_camera'].camera
                    are_visible = self.check_objects_visible(obj_start, obj_end, camera)
                    #only for grasp task, temporarily setting the are_visible = true (to change)
                    are_visible = True
                    if are_visible:
                        break
                    self.initalize_render_camera()
                    self._setup_sensors(options)
                if not are_visible:
                    print("Warning: could not sample visible camera position.")

    def set_goal_pose(self, objA_goal_pose):
        # Move cubeA onto cubeB
        assert self.cubeA is not None
        self.objA_goal_pose = objA_goal_pose
        self.objA_to_goal_dist_inital = torch.linalg.norm(self.cubeA.pose.p - objA_goal_pose.p, axis=1)

    # For the old rewards see: ../mani_skill/envs/tasks/tabletop/stack_cube.py
    def eval_reward(self):
        assert self.cubeA is not None
        objA_pose = self.cubeA.pose.p
        objA_to_goal_dist = torch.linalg.norm(objA_pose - self.objA_goal_pose.p, axis=1)
        reward = torch.clamp(1 - objA_to_goal_dist / self.objA_to_goal_dist_inital, 0, 1)
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.eval_reward()

    def _get_obs_extra(self, info: Dict):
        assert self.agent is not None
        tcp_pose = self.agent.tcp.pose
        robot_pose = self.agent.robot.get_root_pose()
        obs = dict(obj_start=self.obj_start.raw_pose, obj_end=self.obj_end.raw_pose,
                   grasp_pose=self.grasp_pose.raw_pose, tcp_pose=tcp_pose.raw_pose, robot_pose=robot_pose.raw_pose)
        return obs

    def get_obs_scene(self, save_actors=True):
        """get scene info"""
        object_info = {}
        for seg_id, obj in sorted(self.unwrapped.segmentation_id_map.items()):
            save_actors_now = save_actors and obj in self.scene.actors.values()
            save_objects = obj in self.objects
            if not (save_actors_now or save_objects):
                continue

            if obj == self.cubeA:
                obj_in_action = 1
            elif obj == self.cubeB:
                obj_in_action = 2
            else:
                obj_in_action = 0

            obj_in_action = int(obj in (self.cubeA, self.cubeB))
            object_info[obj.name] = dict(seg_id=seg_id, task_req=obj_in_action)
            try:
                # seg_id_to_initial_frame_percent written in run_env.py
                seg_percent = self.seg_id_to_initial_frame_percent[obj.name]
                object_info[obj.name]["seg_percent"] = seg_percent
            except (AttributeError, KeyError):
                pass

        scene_info = dict(text=self.action_text, object_info=object_info)
        return scene_info

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
        return super().reset(seed=seed, options=options)
