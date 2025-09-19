"""
Minimal example script for converting a dataset to LeRobot format.

Usage:
uv run examples/rlbench/convert_rlbench_data_to_lerobot.py --data_dir /path/to/your/data

The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import os
import numpy as np

REPO_NAME = "gaystarc/rlbench_12tasks"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
   "close_box",
   "close_fridge",
   "close_laptop_lid",
   "phone_on_base",
   "sweep_to_dustpan",
   "take_frame_off_hanger",
   "take_umbrella_out_of_umbrella_stand",
   "toilet_seat_down",
   "lamp_on",
   "place_wine_at_rack_location",
   "unplug_charger",
   "water_plants",
]  # Task names of the raw RLBench datasets


def main(data_dir: str, REPO_NAME: str):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    print(output_path)
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
            "image": {
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

    # Loop over raw RLBench npys and write episodes to the LeRobot dataset
    for raw_dataset_name in RAW_DATASET_NAMES:
        print(raw_dataset_name)
        task_dir = os.path.join(data_dir, raw_dataset_name)
        for root, dirs, files in os.walk(task_dir):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    episode = np.load(file_path,allow_pickle=True)

                for i, step in enumerate(episode):
                    image = step['front_image']
                    state = step['state'].astype(np.float32)
                    action = step['action'].astype(np.float32)
                    # image = step['image']
                    # state = step['pose_state'].astype(np.float32)
            
                    dataset.add_frame({
                        "image": image,
                        "state": state,
                        "actions": action,
                        "task": step["language_instruction"]
                    })
                dataset.save_episode()

    # Consolidate the dataset, skip computing stats since we will do that later

if __name__ == "__main__":
    tyro.cli(main)
