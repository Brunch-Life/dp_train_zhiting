
cd /ML-vePFS/tangyinzhou/yinuo/dp_train_zhiting
/ML-vePFS/zhangxin/envs/mininconda/envs/dp_train/bin/python scripts/train.py \
--data_root dataset/processed/nonstop_plate_wooden_new \
--batch_size 128 \
--lr 1e-5 \
--train_seed 0 \
--num_steps 200_000 \
--save_every 5_000 \
--validate_every 2_000 \
--chunk_size 20 \
--total_episode_num 500 