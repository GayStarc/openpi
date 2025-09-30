import shutil
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import os
import numpy as np
from PIL import Image
from typing import List
import glob

def collect_npy_files(data_dirs: List[str]) -> List[str]:
    """收集所有指定文件夹下的npy文件路径"""
    npy_files = []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Warning: Directory {data_dir} does not exist, skipping...")
            continue
            
        # 使用glob递归搜索所有npy文件
        pattern = os.path.join(data_dir, "**", "*.npy")
        files_in_dir = glob.glob(pattern, recursive=True)
        npy_files.extend(files_in_dir)
        print(f"Found {len(files_in_dir)} npy files in {data_dir}")
    
    print(f"Total {len(npy_files)} npy files collected")
    return npy_files

def safe_load_episode(file_path: str):
    """安全加载npy文件"""
    try:
        episode = np.load(file_path, allow_pickle=True)
        return episode
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def main(data_dirs: List[str], REPO_NAME: str, TASK_PROMPT: str):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    print(f"Output path: {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset
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
            "image_wrist_right": {
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

    # 收集所有npy文件路径
    npy_file_list = collect_npy_files(data_dirs)
    
    if not npy_file_list:
        print("No npy files found in the specified directories!")
        return

    # 从文件列表中加载数据集
    successful_episodes = 0
    failed_episodes = 0
    
    for file_path in npy_file_list:
        print(f"Processing: {file_path}")
        
        # 安全加载episode
        episode = safe_load_episode(file_path)
        if episode is None:
            failed_episodes += 1
            continue
            
        try:
            # 处理episode中的每个step
            for i, step in enumerate(episode):
                try:
                    # front_image = Image.fromarray(step['image_third'])
                    # wrist_image = Image.fromarray(step['image_wrist_left'])
                    # right_image = Image.fromarray(step['image_wrist_right'])

                    target_image = Image.fromarray(step['front_goal'])

                    front_image = Image.fromarray(step['front_image'])
                    wrist_image = Image.fromarray(step['left_image'])
                    right_image = Image.fromarray(step['right_image'])

                    task = TASK_PROMPT
                    
                    dataset.add_frame({
                        "image_front": front_image,
                        "image_wrist": wrist_image,
                        "image_wrist_right": right_image,
                        "target_keyframe_image": target_image,
                        "state": step["state"].astype(np.float32),
                        "actions": step["action"].astype(np.float32),
                        "task": task,
                    })
                except Exception as e:
                    print(f"Error processing step {i} in {file_path}: {e}")
                    continue
                    
            dataset.save_episode()
            successful_episodes += 1
            # print(f"Successfully processed episode from {file_path}")
            
        except Exception as e:
            print(f"Error processing episode from {file_path}: {e}")
            failed_episodes += 1
            continue
    
    print(f"\nProcessing completed:")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Failed episodes: {failed_episodes}")
    print(f"Total files processed: {len(npy_file_list)}")

if __name__ == "__main__":
    tyro.cli(main)