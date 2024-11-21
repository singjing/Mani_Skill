from typing import List

import numpy as np
from mani_skill.utils.structs import Pose

from mani_skill.examples.motionplanning.panda.utils import get_actor_obb

# Text Generation stuff

def create_text_names(env):
    objects_descr = env.base_env.objects_descr
    names = []
    for descr in objects_descr:
        name = f'{descr["size"]} {descr["color"]} {descr["shape"]}'
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
    objects = env.base_env.objects
    assert len(objects) >= 2
    object_id_move, object_id_base = np.random.choice(range(len(objects)), 2, replace=False)
    pose_base = objects[object_id_base].pose
    start_pose = objects[object_id_move].pose
    # Pose.create creates a reference
    end_pose = Pose.create_from_pq(p=start_pose.get_p(), q=start_pose.get_q())    
    pos_new = start_pose.get_p().clone().detach()  # don't forget
    pos_base = pose_base.get_p()
    pos_new[:, 0:2] = pos_base[:, 0:2]
    
    obbB = get_actor_obb(env.unwrapped.cubeB, to_world_frame=True)
    obbA = get_actor_obb(env.unwrapped.cubeA, to_world_frame=True)
    height = pos_base[:, 2] + obbB.primitive.extents[2]/2 + obbA.primitive.extents[2]/2
    height = obbB.vertices[:, 2].max() + (obbA.vertices[:, 2].max() - obbA.vertices[:, 2].min())/2
    #from pdb import set_trace
    #set_trace()
    old_height = pos_new[:, 2] + 2*pos_base[:, 2]
    #print(">>>", old_height, height)
    pos_new[:, 2] = height
    end_pose.set_p(pos_new)

    if not pretend:
        objects[object_id_move].set_pose(end_pose)
    
    env.base_env.cubeA = objects[object_id_move]
    env.base_env.cubeB = objects[object_id_base]
    
    # now creat a text
    text_names = create_text_names(env)
    verbs = ["move", "place", "transfer", "set", "position", "lift", "relocate", "shift", "put", "bring"]
    prepositions = ["onto", "on top of", "above", "over", "to rest on", "on", "onto the surface of"]
    if randomize_text:
        verb = np.random.choice(verbs)
        prep = np.random.choice(prepositions)
    else:
        verb = verbs[0]
        prep = prepositions[0]
    action_text = f"{verb} {text_names[object_id_move]} {prep} {text_names[object_id_base]}"
    return start_pose, end_pose, action_text


# TODO(shardul):
# Option1: start here
# move_object_next_to
# move_object_leftrightbehind
# move_object_upright (select shapes)
# move_object_between
# upside down
# rotate
