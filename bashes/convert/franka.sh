export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/guchenyang/Code/HuggingFace
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
export WANDB_API_KEY=21a8bda930a645b08f2834efd21ba21e98cd83cf

uv run examples/franka_dual/convert_franka_data_to_lerobot_dev.py \
  --data-dirs \
  "/home/guchenyang/Code/Data/Franka/Processed/0925_Demo3_npy_key" \
  --REPO-NAME gaystarc/0930_franka_dual_target_image_keyframe \

uv run scripts/compute_norm_stats.py --config-name pi0_franka_dual_dev_lora

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_franka_dual --exp-name=0918_pi05_franka_dual_robomind_task1 --overwrite