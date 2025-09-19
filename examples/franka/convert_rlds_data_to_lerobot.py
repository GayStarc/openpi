import shutil
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

def main(data_dir: str, REPO_NAME: str):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image_front": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "image_wrist": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],  
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Franka npys and write episodes to the LeRobot dataset
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                episode = np.load(file_path, allow_pickle=True)

            for i, step in enumerate(episode):
                dataset.add_frame({
                    "image_front": step['front_image'],
                    "image_wrist": step['wrist_image'],
                    "state": step["state"],
                    "actions": step["action"],
                })
                
            dataset.save_episode(task=step["language_instruction"])

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)


if __name__ == "__main__":
    tyro.cli(main)
