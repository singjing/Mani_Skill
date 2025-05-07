import os
from pathlib import Path
from typing import Annotated
from datetime import datetime
from dataclasses import dataclass

import tyro
import h5py
from tqdm import tqdm


def get_latest_h5(directory):
    """Find the most recent .h5 file in the given directory."""
    h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    if not h5_files:
        return []  # No .h5 file in directory
    # Extract timestamps and sort
    h5_files.sort(key=lambda x: datetime.strptime(x.split('.')[0], "%Y%m%d_%H%M%S"))
    return [Path(f) for f in h5_files]


def merge_h5_files(directories, output_file):
    """Merge the latest .h5 files from each directory into one, maintaining traj order."""
    all_trajs = []
    traj_offset = 0  # To ensure continuous indexing    
    with h5py.File(output_file, 'w') as out_h5:
        for directory in tqdm(directories, desc="dirs", total=len(directories)):
            for h5_fn in tqdm(get_latest_h5(directory)):
                latest_h5 = Path(directory) / h5_fn
                if not latest_h5:
                    print(f"Skipping {directory}, no .h5 file found.")
                    continue
                with h5py.File(latest_h5, 'r') as h5_file:
                    traj_keys = sorted(h5_file.keys(), key=lambda k: int(k.split('_')[-1]))  # Sort traj_0, traj_1, ...
                    for key in traj_keys:
                        new_traj_key = f"traj_{traj_offset}"
                        h5_file.copy(key, out_h5, name=new_traj_key)
                        traj_offset += 1  # Increment to maintain unique indices
    
    print(f"Merged {traj_offset} trajectories into {output_file}")


@dataclass
class Args:
    root_dir: Annotated[str, tyro.conf.arg(aliases=["--root_dir"])] = "/tmp/cvla-1"
    """The root directory to merge"""

    datasets_dir: Annotated[str, tyro.conf.arg(aliases=["--datasets_dir"])] = "/data/lmbraid19/argusm/datasets"
    """Where to put the merged data"""
    

if __name__ == "__main__":
    import shutil
    import subprocess
    # Get all pN directories in the current directory
    parsed_args = tyro.cli(Args)
    root_dir = Path(parsed_args.root_dir)
    assert root_dir.exists(), f"Directory {root_dir} does not exist."
    print("source:", root_dir)

    print("\ncomponets:")
    p_dirs = sorted([root_dir /  d for d in root_dir.iterdir() if (root_dir /  d).is_dir()])    
    for i in p_dirs:
        print("\t", i)
    
    reference_h5 = get_latest_h5(p_dirs[0])[0]
    out_fn = root_dir / reference_h5.name
    print("\n5_file", out_fn)

    # start doing
    source_json = p_dirs[0] / reference_h5.with_suffix('.json')
    dest_json = out_fn.with_suffix('.json')
    print("json_file", dest_json)
    shutil.copyfile(source_json, dest_json)  # copy json file to destination
    
    # Merge into output.h5
    merge_h5_files(p_dirs, out_fn)

    
    old_dir = Path(root_dir)
    old_dir = root_dir.with_name(root_dir.name + '_old')
    print("moving " + str(p_dirs[0]) + " to " + str(old_dir), '...')
    for p in tqdm(p_dirs):
        shutil.move(p, old_dir / p.name)
    
    datasets_dir = parsed_args.datasets_dir
    cmd = f"rsync -a --progress {root_dir} {datasets_dir}"
    print("\n" + cmd)
    subprocess.run(cmd, shell=True)
    print("done.")
    