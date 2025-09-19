export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/gpfs/0607-cluster/guchenyang/huggingface

uv run examples/rlbench/convert_rlbench_data_to_lerobot.py --REPO_NAME gaystarc/rlbench_12tasks_keyframe_10 --data_dir /gpfs/0607-cluster/guchenyang/Data/RLBench/Process/0912_keyframe_10/for_rlds
uv run scripts/compute_norm_stats.py --config-name pi05_rlbench