export CUDA_VISIBLE_DEVICES=0
uv run examples/agilex/convert_agilex_data_to_lerobot.py --raw_dir /home/gcy/agilex_data/pick_and_place_new --repo_id gaystarc/agilex_pick_place_0225_10 --task "Pick up the two objects and place them in the basket. " --use-joint --step 10

uv run examples/agilex/convert_agilex_data_to_lerobot_key.py --raw_dir /home/gcy/agilex_data/pick_place_0301 --repo_id gaystarc/agilex_pick_place_key_3_0332101 --task "Pick up the two objects and place them in the basket. " --no-use-joint --num 3

uv run examples/agilex/convert_agilex_data_to_lerobot_key.py --raw_dir /home/gcy/agilex_data/lift_and_place_0226/test_compressed --repo_id gaystarc/agilex_lift_place_key_1_02272313 --task "Lift up the ball and place it in the basket. " --no-use-joint --num 1
uv run scripts/compute_norm_stats.py --config-name pi0_agilex

uv run examples/agilex/convert_agilex_data_to_lerobot_key.py --raw_dir /home/gcy/agilex_data/place_bottle_rack_0228 --repo_id gaystarc/agilex_place_bottle_rack_0332301 --task "Grasp the two beer bottles and place them on the rack. " --no-use-joint --num 3

uv run examples/agilex/convert_agilex_data_to_lerobot_key.py --raw_dir /home/gcy/agilex_data/place_bottle_rack_0228 --repo_id gaystarc/agilex_place_bottle_rack_0303 --task "Grasp the two wine bottles and place them on the rack. " --no-use-joint --num 3

export CUDA_VISIBLE_DEVICES=0
uv run examples/agilex/convert_agilex_data_to_lerobot_key.py --raw_dir /home/gcy/agilex_data/wipe_board_0303 --repo_id gaystarc/agilex_wipe_board_key_3_0305_ttt --task "Grasp the eraser and wipe out the color on the whiteboard. " --no-use-joint --num 3

export CUDA_VISIBLE_DEVICES=0
uv run examples/agilex/convert_agilex_data_to_lerobot_key.py --raw_dir /home/gcy/agilex_data/fold_shorts --repo_id gaystarc/agilex_fold_shorts_key_3_0304_ttt --task "Fold the shorts into a rectangle. " --no-use-joint --num 3
uv run scripts/compute_norm_stats.py --config-name pi0_agilex