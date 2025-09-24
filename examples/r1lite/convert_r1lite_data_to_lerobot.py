import shutil
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import os
import numpy as np
from PIL import Image

def main(data_dir: str, REPO_NAME: str):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    print(f"Output path: {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="r1lite",
        fps=10,
        features={
            "image_head": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "image_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],  
            },
            "image_right": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],  
            },
            "state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Franka npys and write episodes to the LeRobot dataset
    all_npy = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                all_npy.append(file_path)

    for npy in all_npy:
        episode = np.load(npy, allow_pickle=True)
        for i, step in enumerate(episode):
            
            dataset.add_frame({
                "image_head": step['image_head'],
                "image_left": step['image_left_wrist'],
                "image_right": step['image_right_wrist'],
                "state": step["state"].astype(np.float32),
                "actions": step["action"].astype(np.float32),
                "task": step["language_instruction"],
            })
            
        dataset.save_episode()



if __name__ == "__main__":
    tyro.cli(main)
