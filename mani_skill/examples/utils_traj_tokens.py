import numpy as np

def traj2d_to_tokenstr(traj, resolution_wh, end_string="1"):
    """convert bbox (xyxy) format into paligemma token string
    Arguments:
        traj: array with shape (N waypoints, 2)"""
    assert traj.shape[1] == 2
    w, h = resolution_wh
    # Convert from resolution_wh back to 1024x1024 space
    traj_1024 = traj / np.array([[w, h]]) * 1024
    traj_1024 = traj_1024.astype(int)
    # Construct the result string using the original pattern
    result = ""
    for keypoint in traj_1024:
        result += f"<loc{keypoint[1]:04d}><loc{keypoint[0]:04d}>"
    result +=  f" {end_string}"
    return result.strip()  # Remove any trailing newlines