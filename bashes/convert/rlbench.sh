export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/gpfs/0607-cluster/guchenyang/huggingface

uv run examples/rlbench/convert_rlbench_data_to_lerobot.py --data_dir /gpfs/0607-cluster/liuzhuoyang/data/rlbench/npy/keyframe_delta_position_abs_euler_1024_nextpc_0806/for_rlds/
uv run scripts/compute_norm_stats.py --config-name pi0_rlbench