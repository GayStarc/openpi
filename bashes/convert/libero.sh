export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PYTHONPATH=/media/guchenyang/openpi:$PYTHONPATH
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /media/guchenyang/HuggingFace/modified_libero_rlds
uv run scripts/compute_norm_stats.py --config-name pi0_libero