export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_VISIBLE_DEVICES=0,1,2,3
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/train.py pi0_franka --exp-name=0223_franka_keyframe_pick_place --overwrite
