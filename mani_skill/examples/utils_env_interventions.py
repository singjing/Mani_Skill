import numpy as np
import sapien.physx as physx
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs import Actor
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh


def get_actor_mesh(actor: Actor, to_world_frame=True, vis=False):
    mesh = get_component_mesh(
        actor._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
        to_world_frame=to_world_frame,
    )
    assert mesh is not None, "can not get actor mesh for {}".format(actor)
    return mesh


def move_object_onto(env, pretend=False):
    # Sample objects
    env = env.unwrapped
    objects = env.objects
    assert len(objects) >= 2
    options = np.where(env.objects_unique)[0]
    if len(options) >= 2:
        object_id_move, object_id_base = np.random.choice(options, 2, replace=False)
    else:
        print("Warning: not enough unique objects")
        object_id_move, object_id_base = np.random.choice(range(len(objects)), 2, replace=False)

    # Move cubeA onto cubeB
    env.cubeA = objects[object_id_move]
    env.cubeB = objects[object_id_base]
    obj_start_pose = objects[object_id_move].pose
    pose_base = objects[object_id_base].pose
    # Pose.create creates a reference
    obj_end_pose = Pose.create_from_pq(p=obj_start_pose.get_p(), q=obj_start_pose.get_q())    
    pos_new = obj_start_pose.get_p().clone().detach()  # don't forget
    pos_base = pose_base.get_p()
    pos_new[:, 0:2] = pos_base[:, 0:2]

    if env.object_dataset == "clevr":
        # primitive shapes have origin in center
        meshA = get_actor_mesh(env.cubeA, to_world_frame=True)
        meshB = get_actor_mesh(env.cubeB, to_world_frame=True)
        height = float(meshB.vertices[:,2].max() + (meshA.vertices[:,2].max() - meshA.vertices[:,2].min())/2)
    else:
        # meshes (should) have origin at bottom
        meshB = get_actor_mesh(env.cubeB, to_world_frame=True)
        height = float(meshB.vertices[:, 2].max())
    
    pos_new[:, 2] = height

    # debug_marker = False
    # if debug_marker:
    #     import sapien
    #     builder = env.scene.create_actor_builder()
    #     builder.add_sphere_visual(pose=sapien.Pose(p=[pos_base[0][0], pos_base[0][1], height]), radius=.02, material=sapien.render.RenderMaterial(base_color=[0., 1., 0., 1.]))
    #     marker_visual = builder.build_kinematic(name="marker_visual")
    
    obj_end_pose.set_p(pos_new)

    if not pretend:
        print("trying to move.")
        objects[object_id_move].set_pose(obj_end_pose)
        
    # now creat a text
    text_names = env.object_names
    verb = "move"
    prep = "onto"
    action_text = f"{verb} {text_names[object_id_move]} {prep} {text_names[object_id_base]}"
    return obj_start_pose, obj_end_pose, action_text


# TODO(shardul):
# Option1: start here
# move_object_next_to
# move_object_leftrightbehind
# move_object_upright (select shapes)
# move_object_between
# upside down
# rotate