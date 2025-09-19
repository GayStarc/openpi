export CUDA_VISIBLE_DEVICES=6
uv run examples/franka/convert_franka_data_to_lerobot.py --data_dir /home/lzy/franka_data/pick_and_place_0221_keyframe_new --REPO_NAME gaystarc/franka_keyframe_pick_place
uv run scripts/compute_norm_stats.py --config-name pi0_franka

uv run examples/rlbench/convert_rlbench_data_to_lerobot.py --data_dir /home/cx/ch_collect_keypoints_rlbench_0223_joint_dense/for_rlds --output_dir openpi_data
uv run scripts/compute_norm_stats.py --config-name pi0_rlbench