import os
import h5py
from datetime import datetime
from tqdm import tqdm

def get_latest_h5(directory):
    """Find the most recent .h5 file in the given directory."""
    h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    if not h5_files:
        return []  # No .h5 file in directory
    # Extract timestamps and sort
    h5_files.sort(key=lambda x: datetime.strptime(x.split('.')[0], "%Y%m%d_%H%M%S"), reverse=True)
    return h5_files


def merge_h5_files(directories, output_file):
    """Merge the latest .h5 files from each directory into one, maintaining traj order."""
    all_trajs = []
    traj_offset = 0  # To ensure continuous indexing    
    with h5py.File(output_file, 'w') as out_h5:
        for directory in tqdm(directories, desc="dirs", total=len(directories)):
            for h5_fn in get_latest_h5(directory):
                latest_h5 = Path(directory) / h5_fn
                if not latest_h5:
                    print(f"Skipping {directory}, no .h5 file found.")
                    continue
                with h5py.File(latest_h5, 'r') as h5_file:
                    traj_keys = sorted(h5_file.keys(), key=lambda k: int(k.split('_')[-1]))  # Sort traj_0, traj_1, ...
                    for key in tqdm(traj_keys, desc="file", total=len(traj_keys)):
                        new_traj_key = f"traj_{traj_offset}"
                        h5_file.copy(key, out_h5, name=new_traj_key)
                        traj_offset += 1  # Increment to maintain unique indices
    
    print(f"Merged {traj_offset} trajectories into {output_file}")

from pathlib import Path

if __name__ == "__main__":
    # Get all pN directories in the current directory
    root_dir = Path("/tmp/cvla-7-obja")
    p_dirs = sorted([root_dir /  d for d in root_dir.iterdir() if (root_dir /  d).is_dir()])
    
    print(p_dirs)

    out_fn = root_dir / Path(get_latest_h5(p_dirs[0])[0]).name
    print(out_fn)

    # Merge into output.h5
    merge_h5_files(p_dirs, out_fn)