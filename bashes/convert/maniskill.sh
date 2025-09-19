export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=/media/guchenyang/openpi:$PYTHONPATH
uv run examples/maniskill/convert_maniskill_data_to_lerobot.py --data_dir /media/guchenyang/Data/maniskill_npy/0725_pi0 --REPO_NAME gaystarc/maniskill_4tasks_0725
uv run scripts/compute_norm_stats.py --config-name pi0_maniskill