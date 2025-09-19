export CUDA_VISIBLE_DEVICES=1
# uv run examples/r1lite/convert_r1lite_data_to_lerobot.py --data_dir /gpfs/0607-cluster/guchenyang/Data/R1LITE/Processed/1000_joint_pick_place_keyframe --REPO_NAME gaystarc/1000_r1lite_pick_place_joint
# uv run scripts/compute_norm_stats.py --config-name pi0_r1lite_joint

uv run examples/r1lite/convert_r1lite_data_to_lerobot.py --data_dir /gpfs/0607-cluster/guchenyang/Data/R1LITE/Processed/0827_pick_place_dense --REPO_NAME gaystarc/0827_r1lite_pick_place_dense
uv run scripts/compute_norm_stats.py --config-name pi0_r1lite_dense