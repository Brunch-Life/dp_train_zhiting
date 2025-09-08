CUDA_VISIBLE_DEVICES=3 python scripts/train.py \
--data_root dataset/processed/nonstop_plate_wooden_new \
--batch_size 128 \
--lr 1e-5 \
--train_seed 0 \
--num_steps 200_000 \
--save_every 5_000 \
--validate_every 5_000 \
--chunk_size 20 \
--total_episode_num 500 \
