from typing import List

import numpy as np
from mani_skill.utils.structs import Pose


# Text Generation stuff

def create_text_names(env):
    objects_descr = env.base_env.objects_descr
    names = []
    for descr in objects_descr:
        name = f'{descr["size"]} {descr["color"]} {descr["shape"]}'
        names.append(name)
    return names


# Environment interactions
def move_object_onto(env, randomize_text=False):
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
    pos_new[:, 2] = pos_new[:, 2] + 2*pos_base[:, 2]
    end_pose.set_p(pos_new)
    objects[object_id_move].set_pose(end_pose)

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
