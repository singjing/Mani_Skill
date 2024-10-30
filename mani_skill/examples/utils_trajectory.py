"""
This file is supposed to have utils for generating trajectories, meaning geometric stuff.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm


def batch_multiply(tensor_A, tensor_B):
    # Reshape tensor_B to (N, 4, 1) to enable batch matrix multiplication
    tensor_B = tensor_B.unsqueeze(-1)
    # Perform batch matrix multiplication
    result = torch.bmm(tensor_A, tensor_B)
    # Remove the last dimension to get shape (N, 3)
    result = result.squeeze(-1)
    return result


def project_point(camera, point_3d):
    """
    Projects a 3D point onto the 2D image plane.

    Parameters:
    - camera: RenderCamera instance
    - point_3d: pytorch tensor of shape (N, 3), I think N batches envs
    
    Returns:
    - A tuple (x, y) representing the 2D image coordinates.
    """
    extrinsic_matrix = camera.get_extrinsic_matrix()
    # Convert the point to homogeneous coordinates
    point_3d_h = torch.cat([point_3d, torch.ones((len(point_3d), 1))], dim=1)    
    # Transform to camera coordinates
    point_camera = batch_multiply(extrinsic_matrix, point_3d_h)
    # Step 2: Get the intrinsic matrix for projection
    intrinsic_matrix = camera.get_intrinsic_matrix()
    # Project onto the image plane
    point_image_h = batch_multiply(intrinsic_matrix, point_camera[:,:3])
    # Normalize by the z-coordinate to get 2D image coordinates
    point_image_n = point_image_h[:, :2] / point_image_h[:, 2]
    return point_image_n


def project_points(camera, points_3d):
    """
    Projects a batch of 3D points onto the 2D image plane.
    
    Parameters:
    - camera: An instance of the Camera class with necessary methods.
    - points_3d: A tensor of shape (N, P, 3) representing the 3D points in world coordinates.
    
    Returns:
    - A tensor of shape (N, P, 2) representing the 2D image coordinates.
    """
    # Step 1: Get the extrinsic matrix and expand for batch processing
    extrinsic_matrix = camera.get_extrinsic_matrix()  # Shape (3, 4)
    extrinsic_matrix = extrinsic_matrix.unsqueeze(0).expand(-1, points_3d.shape[1], -1, -1) # shape (N, P, 3, 4)
    # Convert points to homogeneous coordinates by adding a fourth dimension of ones
    ones = torch.ones((*points_3d.shape[:2], 1))  # Shape (N, P, 1)
    points_3d_h = torch.cat([points_3d, ones], dim=-1)  # Shape (N, P, 4)
    # Transform to camera coordinates using the extrinsic matrix
    points_camera = batch_multiply(extrinsic_matrix.view(-1, 3, 4),points_3d_h.view(-1,4))
    # Step 2: Get the intrinsic matrix and apply it for projection
    intrinsic_matrix = camera.get_intrinsic_matrix()  # Shape (N, 3, 3)
    intrinsic_matrix = intrinsic_matrix.unsqueeze(0).expand(-1, points_3d.shape[1], -1, -1) # shape (N, P, 3, 4)    
    # Project onto the image plane by applying the intrinsic matrix
    points_image_h = batch_multiply(intrinsic_matrix.view(-1, 3, 3), points_camera.view(-1, 3))
    points_image_h = points_image_h.view(points_3d.shape[0], points_3d.shape[1], 3)  # Shape (N, P, 3)
    # Normalize by the z-coordinate to get 2D image coordinates
    x = points_image_h[..., 0] / points_image_h[..., 2]
    y = points_image_h[..., 1] / points_image_h[..., 2]
    return torch.stack((x, y), dim=-1)  # Shape (N, P, 2)


def generate_curve_torch(points_a, points_b, up_dir=(0, 0, 1), height_scale=1, num_points=20):
    """
    Generate a Bézier going from points_a to points_b via points_mid
    Arguments:
        points_a: tensor shape (N, 3)
    """
    assert points_a.ndim == 2 and points_b.ndim == 2
    assert points_a.shape[1] == 3 and points_a.shape[1] == 3
    assert up_dir == (0, 0, 1)
    assert torch.is_tensor(points_a)
    assert torch.is_tensor(points_b)
    distance_between = torch.linalg.norm(points_a - points_b, axis=1)
    mid_point = (points_a + points_b)/2
    points_mid = mid_point + torch.tensor(up_dir)*distance_between*height_scale
    t = torch.linspace(0, 1, num_points).view(1, -1, 1)  # gives shape (1, num_points, 1)
    # # Calculate the Bézier curve points
    curve = (1-t)**2*points_a.unsqueeze(1) + 2*(1-t)*t*points_mid.unsqueeze(1) + t**2*points_b.unsqueeze(1)
    assert curve.shape[1:] == (num_points, 3)
    return (points_a, points_mid, points_b), curve


def subsample_trajectory(trajectory_old, points_new=8):
    """
    Subsamples a trajectory of shape (N, 20, 2) to a new shape (B, points_new, 2).

    Parameters:
    - trajectory_old: A NumPy array of shape (N, 20, 2) representing the old trajectories.
    - N: The number of points to sample from each trajectory (default is 8).

    Returns:
    - A NumPy array of shape (N, points_new, 2) representing the subsampled trajectories.
    """
    N, points_old, _ = trajectory_old.shape
    x_old = np.linspace(0, points_old - 1, points_old)
    x_new = np.linspace(0, points_old - 1, points_new)
    interpolated_array = np.zeros((N, points_new, 2))
    for i in range(trajectory_old.shape[0]):
        for j in range(trajectory_old.shape[2]):
            interpolated_array[i, :, j] = np.interp(x_new, x_old, trajectory_old[i, :, j])
    return interpolated_array

# Plotting
def plot_gradient_curve(axs, x, y, colormap='viridis'):
    """
    Plots a curve with colors progressing along its length.

    Parameters:
    - axs: The Matplotlib axis object to plot on.
    - x: The x-coordinates of the curve.
    - y: The y-coordinates of the curve.
    - colormap: The colormap to use for the progression (default is 'viridis').
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap=colormap, norm=norm)
    lc.set_array(np.linspace(0, 1, len(segments)))  # Gradient progression along the curve
    axs.add_collection(lc)

