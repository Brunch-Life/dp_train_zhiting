
cd /ML-vePFS/tangyinzhou/yinuo/dp_train_zhiting
/ML-vePFS/zhangxin/envs/mininconda/envs/dp_train/bin/python datasets/convert_bingwen.py \
--root_dir "/ML-vePFS/tangyinzhou/bingwen/ManiSkill/videos/datasets_mp/TabletopPickPlaceEnv-v1/20250905_114601" \
--save_dir "dataset/processed/nonstop_plate_wooden_new" \
--tasks-dir nonstop_plate_wooden \
--max_task_num 1