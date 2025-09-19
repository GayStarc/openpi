export https_proxy=http://127.0.0.1:7893 http_proxy=http://127.0.0.1:7893
export CUDA_HOME=/usr/local/cuda-12.4
# export CUDA_VISIBLE_DEVICES=0
# uv run examples/agilex/convert_agilex_data_to_lerobot.py --raw_dir /home/gcy/agilex_data/pick_and_place --repo_id gaystarc/agilex_pose --task "Pick up the two objects and place them in the basket. " --no-use-joint

export CUDA_VISIBLE_DEVICES=0,1,2,3
XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 uv run scripts/train.py pi0_agilex --exp-name=0224_agilex_pose_pick_place --overwrite
