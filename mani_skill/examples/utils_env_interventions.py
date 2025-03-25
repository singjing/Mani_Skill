from typing import List

import numpy as np
from mani_skill.utils.structs import Pose

from mani_skill.examples.motionplanning.panda.utils import get_actor_obb

# Text Generation stuff

def create_text_names(env):
    objects_descr = env.objects_descr
    names = []
    for descr in objects_descr:
        name = f'{descr["size"]} {descr["color"]} {descr["shape"]}'.strip()
        names.append(name)
    return names


# Environment interactions

# def move_object(env, delta_pos=[0, 0, 0.1]):
#     scene = env.base_env.scene
#     object = scene.get_all_actors()[1]
#     pose = object.pose
#     pose.set_p(pose.p + delta_pos)
#     object.set_pose(pose)

def move_object_onto(env, randomize_text=False, pretend=False):
    # Move cubeA onto cubeB
    env = env.unwrapped
    objects = env.objects
    assert len(objects) >= 2
    options = np.where(env.objects_unique)[0]
    if len(options) >= 2:
        object_id_move, object_id_base = np.random.choice(options, 2, replace=False)
    else:
        print("Warning: not enough unique objects")
        object_id_move, object_id_base = np.random.choice(range(len(objects)), 2, replace=False)
    pose_base = objects[object_id_base].pose
    obj_start_pose = objects[object_id_move].pose
    # Pose.create creates a reference
    obj_end_pose = Pose.create_from_pq(p=obj_start_pose.get_p(), q=obj_start_pose.get_q())    
    pos_new = obj_start_pose.get_p().clone().detach()  # don't forget
    pos_base = pose_base.get_p()
    pos_new[:, 0:2] = pos_base[:, 0:2]

    if env.cubeB.name.startswith("clevr"):
        obbB = get_actor_obb(env.cubeB, to_world_frame=True)
        obbA = get_actor_obb(env.cubeA, to_world_frame=True)
        height = obbB.vertices[:, 2].max() + (obbA.vertices[:, 2].max() - obbA.vertices[:, 2].min())/2
        #height = float(pos_base[:, 2] + obbB.primitive.extents[2]/2 + obbA.primitive.extents[2]/2)
        pos_new[:, 2] = height

    else:
        obbB = get_actor_obb(env.cubeB, to_world_frame=True)
        obbA = get_actor_obb(env.cubeA, to_world_frame=True)
        height = float(pos_base[:, 2] + obbB.primitive.extents[2]/2 + obbA.primitive.extents[2]/2)
        pos_new[:, 2] = height

    debug_marker = False
    if debug_marker:
        import sapien
        builder = env.scene.create_actor_builder()
        builder.add_sphere_visual(pose=sapien.Pose(p=[pos_base[0][0], pos_base[0][1], height]), radius=.02, material=sapien.render.RenderMaterial(base_color=[0., 1., 0., 1.]))
        marker_visual = builder.build_kinematic(name="marker_visual")
    
    obj_end_pose.set_p(pos_new)

    if not pretend:
        print("trying to move.")
        objects[object_id_move].set_pose(obj_end_pose)
    
    env.cubeA = objects[object_id_move]
    env.cubeB = objects[object_id_base]
    
    # now creat a text
    text_names = create_text_names(env)
    verbs = ["move", "place", "transfer", "set", "position", "lift", "relocate", "shift", "put", "bring"]
    prepositions = ["onto", "on top of", "above", "over", "to rest on", "on", "onto the surface of"]
    if randomize_text:
        raise ValueError("Deprecated, do this in data loader.")
        verb = np.random.choice(verbs)
        prep = np.random.choice(prepositions)
    else:
        verb = verbs[0]
        prep = prepositions[0]
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


# AVA - atomic visual actions dataset
# Carry/hold (100598)
# Touch (21099)
# Lift/pick up (634)
# Put down (653)
# Open (1547)
# Close (986)
# Push (465)
# Pull (460)
# Throw (336)
# Catch (97)