from pathlib import Path
import json
import random
import shutil

from pdb import set_trace


def split_dataset(dataset_path: Path, train_ratio: float = 0.8, seed: int = 42):
    """
    Randomly split a JSONL dataset into train and test sets.

    Args:
        dataset_path (Path): Path to the JSONL file.
        train_ratio (float): The ratio of the data to use for training.
        seed (int): Random seed for reproducibility.
    """

    # Define paths for train and test files
    json_file = dataset_path / "_annotations.all.jsonl"
    train_path = dataset_path / "_annotations.train.jsonl"
    test_path = dataset_path / "_annotations.test.jsonl"

    # Load the dataset
    with open(json_file, "r") as file:
        lines = file.readlines()

    # check that all files are there
    refered_files = []    
    for line in lines:
        label = json.loads(line)
        refered_files.append(label["image"])
    print("loading all .jpg images in", dataset_path)
    present_files = list(dataset_path.glob("*.jpg"))
    refered_files_set = set(refered_files)
    present_files_set = {file.name for file in present_files}
    missing_files = refered_files_set - present_files_set
    extra_files = present_files_set - refered_files_set
    print("referred but not present:", len(missing_files))
    print("present but not referred:", len(extra_files))
    assert len(missing_files) == 0

    # Set the random seed for reproducibility
    random.seed(seed)
    # Shuffle and split the dataset
    random.shuffle(lines)
    split_index = int(len(lines) * train_ratio)
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]

    # Write to the train and test files
    with open(train_path, "w") as train_file:
        train_file.writelines(train_lines)
    with open(test_path, "w") as test_file:
        test_file.writelines(test_lines)
    print("train images", len(train_lines))
    print("test  images", len(test_lines))
    
    # create a zip of dataset_path.parent and place it in dataset_path.parent.parent
    print("\nCreating zip, this may take a while...")
    print(dataset_path.parent)
    source_dir = dataset_path.parent
    destination_dir = dataset_path.parent.parent / f"{source_dir.name}.zip"
    zip_path = shutil.make_archive(base_name=destination_dir.with_suffix(''), format='zip', root_dir=source_dir)
    print(f"Created zip at {zip_path} with size: {Path(zip_path).stat().st_size / (1024 * 1024):.2f} MB and {len(lines)} samples")
    print("done.")

if __name__ == "__main__":
    dataset_path = Path("/tmp/clevr-act-1/dataset")
    split_dataset(dataset_path)