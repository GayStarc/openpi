export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
export WANDB_API_KEY=21a8bda930a645b08f2834efd21ba21e98cd83cf

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_franka --exp-name=0825_franka_press_stamp --overwrite
