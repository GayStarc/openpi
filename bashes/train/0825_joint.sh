export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
export WANDB_API_KEY=TBD

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_r1lite_joint --exp-name=1000_r1lite_pick_place_joint --overwrite