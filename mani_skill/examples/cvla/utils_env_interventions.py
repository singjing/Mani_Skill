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


<<<<<<< HEAD
# TODO(shardul):
# Option1: start here
# move_object_next_to
# move_object_leftrightbehind
# move_object_upright (select shapes)
# move_object_between
# upside down
# rotate
=======
def quat_multiply(q1, q2):
    """Computes the Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def move_object_next_to(env, pretend=False, offset=0.1):
    # Move cubeA next to cubeB
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

    # Create a new pose for the object to be moved
    obj_end_pose = Pose.create_from_pq(p=obj_start_pose.get_p(), q=obj_start_pose.get_q())

    # Get the position of the base object
    pos_base = pose_base.get_p()
    pos_start = obj_start_pose.get_p()

    direction = pos_base - pos_start
    direction_normalized = direction / np.linalg.norm(direction)

    # Calculate the new position for the moved object
    pos_new = obj_start_pose.get_p().clone().detach()
    pos_new = pos_base - direction_normalized * offset

    # Set the new position in the end pose
    obj_end_pose.set_p(pos_new)

    if not pretend:
        print("trying to move.")
        objects[object_id_move].set_pose(obj_end_pose)

    env.cubeA = objects[object_id_move]
    env.cubeB = objects[object_id_base]

    # now creat a text
    text_names = env.object_names
    verb = "move"
    prep = "next to"
    action_text = f"{verb} {text_names[object_id_move]} {prep} {text_names[object_id_base]}"

    return obj_start_pose, obj_end_pose, action_text


def move_object_leftrightbehind(env, direction="left", pretend=False, offset=0.1):
    env = env.unwrapped
    objects = env.objects
    assert len(objects) >= 2
    options = np.where(env.objects_unique)[0]
    if len(options) >= 2:
        object_id_move, object_id_base = np.random.choice(options, 2, replace=False)
    else:
        print("Warning: not enough unique objects")
        object_id_move, object_id_base = np.random.choice(range(len(objects)), 2, replace=False)


    camera_config = env.render_camera_config
    output = parse_camera_configs(camera_config)
    camera_pose = output["render_camera"].pose

    # Get the poses of the base object and the object to move
    pose_base = objects[object_id_base].pose
    obj_start_pose = objects[object_id_move].pose

    # Create a copy of the object's pose for the end position
    obj_end_pose = Pose.create_from_pq(p=obj_start_pose.get_p(), q=obj_start_pose.get_q())


    # Compute the direction from the camera to the base object
    base_position = pose_base.get_p()
    cam_pos = camera_pose.get_p()

    adjusted_cam_pos = cam_pos.clone()
    adjusted_cam_pos [:, 2] = base_position [:, 2]

    direction_vector = base_position - adjusted_cam_pos

    # Normalize the flattened direction vector
    direction_unit = direction_vector / torch.norm(direction_vector, dim=1, keepdim=True)

    # Compute right direction (perpendicular vector in XY plane)
    up_vector = torch.zeros_like(direction_unit)
    up_vector[..., 2] = 1

    right_vector = torch.cross(direction_vector, up_vector)
    right_length = torch.norm(right_vector, dim=-1, keepdim=True)
    right_unit = right_vector / (right_length + 1e-6)

    # Compute the new position based on the direction parameter
    if direction == "behind":
        new_position = base_position + offset * direction_unit
    elif direction == "right":
        new_position = base_position + offset * right_unit
    elif direction == "left":
        new_position = base_position - offset * right_unit
    else:
        raise ValueError("Invalid direction. Choose 'left', 'right', or 'behind'.")

    # #Add vertical offset to not go into the table too much, inelegant solution, change later
    # new_position[:, 2] += 0.05

    # Update the object's position to the new position
    pos_new = obj_start_pose.get_p().clone().detach()
    pos_new[:, :3] = new_position
    obj_end_pose.set_p(pos_new)


    if not pretend:
        print("trying to move.")
        objects[object_id_move].set_pose(obj_end_pose)

    env.cubeA = objects[object_id_move]
    env.cubeB = objects[object_id_base]


    # now create a text
    text_names = env.object_names
    verb = "move"
    prep = {
        "behind" : "behind",
        "left" : "to the left of",
        "right" : "to the right of"
        }
    prepfinal = prep[direction]
    action_text = f"{verb} {text_names[object_id_move]} {prepfinal} {text_names[object_id_base]}"
    print(action_text)
    return obj_start_pose, obj_end_pose, action_text

def move_object_between(env, pretend=False):
    # Move cubeA next to cubeB
    env = env.unwrapped
    objects = env.objects
    assert len(objects) >= 3


    options = np.where(env.objects_unique)[0]
    if len(options) >= 3:
        object_id_move, object_id_base_1, object_id_base_2 = np.random.choice(options, 3, replace=False)
    else:
        print("Warning: not enough unique objects")
        object_id_move, object_id_base_1, object_id_base_2 = np.random.choice(range(len(objects)), 3, replace=False)

    pose_base_1 = objects[object_id_base_1].pose
    pose_base_2 = objects[object_id_base_2].pose
    obj_start_pose = objects[object_id_move].pose

    # Create a new pose for the object to be moved
    obj_end_pose = Pose.create_from_pq(p=obj_start_pose.get_p(), q=obj_start_pose.get_q())

    # Get the position of the base object
    pos_base_1 = pose_base_1.get_p()
    pos_base_2 = pose_base_2.get_p()

    # Calculate the new position for the moved object
    pos_new = obj_start_pose.get_p().clone().detach()

    # Place the moved object between the 2 base objects
    midpoint = (pos_base_1 + pos_base_2) / 2

    pos_new[:, :3] = midpoint

    # Set the new position in the end pose
    obj_end_pose.set_p(pos_new)

    if not pretend:
        print("trying to move.")
        objects[object_id_move].set_pose(obj_end_pose)

    env.cubeA = objects[object_id_move]
    env.cubeB = objects[object_id_base_1]
    env.cubeC = objects[object_id_base_2]

    # now creat a text
    text_names = env.object_names
    verb = "place"
    prep = "between"

    action_text = f"{verb} {text_names[object_id_move]} {prep} {text_names[object_id_base_1]} and {text_names[object_id_base_2]}"

    return obj_start_pose, obj_end_pose, action_text

def move_object_rotate_x(env, pretend=False, rotation = "anticlockwise", rotation_angle_deg=120):
    # Move cubeA and rotate it along the x-axis by `rotation_angle_deg`
    env = env.unwrapped
    objects = env.objects
    assert len(objects) >= 1

    options = np.where(env.objects_unique)[0]
    if len(options) >= 1:
        object_id_move = np.random.choice(options, 1, replace=False)[0]
    else:
        print("Warning: not enough unique objects")
        object_id_move = np.random.choice(range(len(objects)), 1, replace=False)[0]
    absolute_rotation = rotation_angle_deg
    if rotation in ["clockwise", "anticlockwise"]:
            if rotation == "clockwise":
                rotation_angle_deg = -1 * rotation_angle_deg
    else:
        print ("Warning : Wrong rotation value, only ""clockwise"" and ""anticlockwise"" are permitted.")


    # Get the pose of the object to move
    obj_start_pose = objects[object_id_move].pose

    # Create a new pose for the object to be moved (initially same as the start pose)
    obj_end_pose = Pose.create_from_pq(p=obj_start_pose.get_p(), q=obj_start_pose.get_q())

    # Get the current orientation (quaternion) of the object
    obj_orientation = obj_end_pose.get_q()

    # Convert the rotation angle to radians
    angle_rad = np.radians(rotation_angle_deg)

    # Create a quaternion for rotation around the X-axis
    q_rotate_x = torch.tensor([np.cos(angle_rad / 2), 0.0, 0.0, np.sin(angle_rad / 2)])

    # Multiply the current orientation by the rotation quaternion
    obj_orientation_rotated = quat_multiply(q_rotate_x, obj_orientation)

    pos_new = obj_start_pose.get_p().clone().detach()
    orientation_new = obj_start_pose.get_q().clone().detach()
    orientation_new = obj_orientation_rotated
    obj_end_pose.set_p(pos_new)
    obj_end_pose.set_q(orientation_new)

    if not pretend:
        print("trying to move.")
        objects[object_id_move].set_pose(obj_end_pose)

    env.cubeA = objects[object_id_move]


    # now creat a text
    text_names = env.object_names
    verb = "rotate"
    adverbs = [f"by {absolute_rotation} degrees {rotation}"]

    action_text = f"{verb} {text_names[object_id_move]} {adverbs[0]}"
    print(action_text)
    return obj_start_pose, obj_end_pose, action_text

def move_object_lr_singular(env, direction="left", pretend=False, offset=0.1):
    env = env.unwrapped
    objects = env.objects
    assert len(objects) >= 1
    options = np.where(env.objects_unique)[0]
    if len(options) >= 1:
        object_id_move = np.random.choice(options, 1, replace=False)[0]
    else:
        print("Warning: not enough unique objects")
        object_id_move = np.random.choice(range(len(objects)), 1, replace=False)[0]


    camera_config = env.render_camera_config
    output = parse_camera_configs(camera_config)
    camera_pose = output["render_camera"].pose

    # Get the poses of the object to move

    obj_start_pose = objects[object_id_move].pose

    # Create a copy of the object's pose for the end position
    obj_end_pose = Pose.create_from_pq(p=obj_start_pose.get_p(), q=obj_start_pose.get_q())


    # Compute the direction from the camera to the base object
    base_position = obj_start_pose.get_p()
    cam_pos = camera_pose.get_p()

    adjusted_cam_pos = cam_pos.clone()
    adjusted_cam_pos [:, 2] = base_position [:, 2]

    direction_vector = base_position - adjusted_cam_pos

    # Normalize the flattened direction vector
    direction_unit = direction_vector / torch.norm(direction_vector, dim=1, keepdim=True)

    # Compute right direction (perpendicular vector in XY plane)
    up_vector = torch.zeros_like(direction_unit)
    up_vector[..., 2] = 1

    right_vector = torch.cross(direction_vector, up_vector)
    right_length = torch.norm(right_vector, dim=-1, keepdim=True)
    right_unit = right_vector / (right_length + 1e-6)

    # Compute the new position based on the direction parameter
    if direction == "right":
        new_position = base_position + offset * right_unit
    elif direction == "left":
        new_position = base_position - offset * right_unit
    else:
        raise ValueError("Invalid direction. Choose 'left' or 'right'.")

    # Update the object's position to the new position
    pos_new = obj_start_pose.get_p().clone().detach()
    pos_new[:, :3] = new_position
    obj_end_pose.set_p(pos_new)

    if not pretend:
        print("trying to move.")
        objects[object_id_move].set_pose(obj_end_pose)

    env.cubeA = objects[object_id_move]


    # Create a text description of the action
    text_names = env.object_names
    verb = "move"
    prepositions = {
        "left": "to the left",
        "right": "to the right"
    }

    prep = prepositions[direction]
    action_text = f"{verb} {text_names[object_id_move]} {prep} "

    return obj_start_pose, obj_end_pose, action_text


def move_object_forward_backward(env, direction="forward", pretend=False, offset=0.1):
    env = env.unwrapped
    objects = env.objects
    assert len(objects) >= 1
    options = np.where(env.objects_unique)[0]
    if len(options) >= 1:
        object_id_move = np.random.choice(options, 1, replace=False)[0]
    else:
        print("Warning: not enough unique objects")
        object_id_move = np.random.choice(range(len(objects)), 1, replace=False)[0]


    camera_config = env.render_camera_config
    output = parse_camera_configs(camera_config)
    camera_pose = output["render_camera"].pose

    # Get the poses of the object to move

    obj_start_pose = objects[object_id_move].pose

    # Create a copy of the object's pose for the end position
    obj_end_pose = Pose.create_from_pq(p=obj_start_pose.get_p(), q=obj_start_pose.get_q())


    # Compute the direction from the camera to the base object
    base_position = obj_start_pose.get_p()
    cam_pos = camera_pose.get_p()

    adjusted_cam_pos = cam_pos.clone()
    adjusted_cam_pos [:, 2] = base_position [:, 2]

    direction_vector = base_position - adjusted_cam_pos

    # Normalize the flattened direction vector
    direction_unit = direction_vector / torch.norm(direction_vector, dim=1, keepdim=True)


    # Compute the new position based on the direction parameter
    if direction == "forward":
        new_position = base_position + offset * direction_unit
    elif direction == "backward":
        new_position = base_position - offset * direction_unit
    else:
        raise ValueError("Invalid direction. Choose 'forward' or 'backward'.")

    # #Add vertical offset to not go into the table too much, inelegant solution, change later
    # new_position[:, 2] += 0.05

    # Update the object's position to the new position
    pos_new = obj_start_pose.get_p().clone().detach()
    pos_new[:, :3] = new_position
    obj_end_pose.set_p(pos_new)

    if not pretend:
        print("Trying to move.")
        objects[object_id_move].set_pose(obj_end_pose)

    env.cubeA = objects[object_id_move]


    # Create a text description of the action
    text_names = env.object_names
    verb = "move"
    prepositions = {
        "forward": "further",
        "backward": "closer"
    }

    prep = prepositions[direction]

    action_text = f"{verb} {text_names[object_id_move]} {prep} "

    return obj_start_pose, obj_end_pose, action_text

>>>>>>> 5b40c960 (fix)
