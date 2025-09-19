export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/gpfs/0607-cluster/guchenyang/huggingface
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
export WANDB_API_KEY=TBD


uv run examples/franka_dual/convert_franka_data_to_lerobot.py \
  --data-dirs \
  "/gpfs/0607-cluster/guchenyang/Data/Franka/Processed/0918_robomind/task2" \
  --REPO-NAME gaystarc/0918_franka_dual_robomind_task2 \

uv run scripts/compute_norm_stats.py --config-name pi05_franka_dual

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_franka_dual --exp-name=0918_pi05_franka_dual_robomind_task2 --overwrite