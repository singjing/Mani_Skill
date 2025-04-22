import os.path as osp
from pathlib import Path
from typing import List, Optional

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.envs.utils import randomization


TABLE_HEIGHT = 0.9196429
table_configs = [
    # default table
    dict(model = "table.glb", p=[-0.12, 0, -TABLE_HEIGHT], q=euler2quat(0, 0, np.pi / 2), q_vis=euler2quat(0, 0, np.pi / 2), scale=1.75),
    # white/grayish table
    dict(model = "box", p=[0, 0, -TABLE_HEIGHT / 2], q=euler2quat(0, 0, 0), color=(200, 200, 200)),
    # tabel model 2
    dict(model = "Dining_Table_204_1.glb", p=[0, 0, -0.64], q=euler2quat(0, 0, 0), q_vis=euler2quat(np.pi/2, 0 , 0), scale=1.0),
    # table model 2 upside down
    dict(model = "Dining_Table_204_1.glb", p=[0, 0, 0.558], q=euler2quat(0, 0, 0), q_vis=euler2quat(-np.pi/2, 0 , 0), scale=1.0),  # upside down
    # random rgb color table
    dict(model = "box", p=[0, 0, -TABLE_HEIGHT / 2], q=euler2quat(0, 0, 0), color="random"),
    ]


class TableSceneBuilder(SceneBuilder):
    NUM_SCENE_OPTIONS = len(table_configs)

    """A simple scene builder that adds a table to the scene such that the height of the table is at 0, and
    gives reasonable initial poses for robots."""
    def build(self, build_config_idxs: Optional[List[int]] = None):
        if build_config_idxs is None:
            build_config_idxs = [0]

        builder = self.scene.create_actor_builder()
        self.table_config = table_configs[build_config_idxs[0]]
        if self.table_config["model"] == "box":
            if self.table_config["color"] == "random":
                base_color = randomization.uniform(0, 1, size=(4,)).cpu().numpy().astype(float)
                base_color[3] = 1.0
            else:
                base_color = [x / 255 for x in self.table_config["color"] + (255,)]

            builder.add_box_visual(half_size=[2.418 / 2, 1.209 / 2, TABLE_HEIGHT / 2],
                                   material=sapien.render.RenderMaterial(base_color=base_color),
            )
            builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, 0.0]),
                half_size=(2.418 / 2, 1.209 / 2, TABLE_HEIGHT / 2),
            )
            builder.initial_pose = sapien.Pose(p=self.table_config["p"])

        else:
            model_dir = Path(osp.dirname(__file__)) / "assets"
            table_model_file = str(model_dir / self.table_config["model"])
            #table_model_file = str(model_dir / "Dining_Table_204_1.glb")
            scale = self.table_config["scale"]
            table_pose = sapien.Pose(q=self.table_config["q_vis"])
            builder.add_visual_from_file(
                filename=table_model_file, scale=[scale] * 3, pose=table_pose
            )
            builder.initial_pose = sapien.Pose(
                p=self.table_config["p"], q=self.table_config["q"]
            )
            # builder.add_nonconvex_collision_from_file(
            #     filename=table_model_file,
            #     scale=[scale] * 3,
            #     pose=table_pose,
            # )

            # initialize set_pose will move up by table_cofig["p"] and we have height
            # TABLE_HEIGHT/2 so undo those to get to 0
            p_collision_box = -1*np.array(self.table_config["p"]) - [0,0,TABLE_HEIGHT/2]
            builder.add_box_collision(
                pose=sapien.Pose(p=p_collision_box),
                half_size=(2.418 / 2, 1.209 / 2, TABLE_HEIGHT / 2),
            )
        table = builder.build_kinematic(name="table-workspace")
        aabb = (
            table._objs[0]
            .find_component_by_type(sapien.render.RenderBodyComponent)
            .compute_global_aabb_tight()
        )
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        self.table = table
        self.scene_objects: List[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        # table_height = TABLE_HEIGHT
        b = len(env_idx)
        #self.table.set_pose(
        #    sapien.Pose(p=[-0.12, 0, -TABLE_HEIGHT], q=euler2quat(0, 0, np.pi / 2))
        #)
        self.table.set_pose(sapien.Pose(p=self.table_config["p"], q=self.table_config["q"]))
        if self.env.robot_uids == "panda":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "panda_wristcam":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in [
            "xarm6_allegro_left",
            "xarm6_allegro_right",
            "xarm6_robotiq",
            "xarm6_nogripper",
        ]:
            qpos = self.env.agent.keyframes["rest"].qpos
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.522, 0, 0]))
        elif self.env.robot_uids == "fetch":
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == ("panda", "panda"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif self.env.robot_uids == ("panda_wristcam", "panda_wristcam"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif (
            "dclaw" in self.env.robot_uids
            or "allegro" in self.env.robot_uids
            or "trifinger" in self.env.robot_uids
        ):
            # Need to specify the robot qpos for each sub-scenes using tensor api
            pass
        elif self.env.robot_uids == "panda_stick":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "so100":
            qpos = np.array([0, np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, 1.0])
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(
                sapien.Pose([-0.725, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        else:
            print("Warning: resetting robot without scene position.")
            self.env.agent.reset()  # add a qpos here for a given robot/scenen combination
            self.env.agent.robot.set_pose(
                sapien.Pose([-0.725, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )
