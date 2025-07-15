### env creation

```bash
pip install -r requirements.txt
pip install numpy==1.23.5 opencv-python==4.11.0.86 sapien==3.0.0b1 huggingface_hub==0.25.2
```


### dataset convert

```bash
PYTHONPATH=. python datasets/convert_bingwen.py \
--root_dir "/tangyinzhou-tos-volc-engine/bingwen/ms_data/TabletopPickPlaceEnv-v1/" \
--save_dir "dataset/processed/nonstop_plate_wooden_1000" \
--tasks-dir nonstop_plate_wooden \
--max_task_num 1
```


### train
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train.py \
--data_root dataset/processed/test_for_new_bingwen_2_task \
--batch_size 128 \
--lr 1e-5 \
--train_seed 0 \
--num_steps 200_000 \
--save_every 20_000 \
--validate_every 2_000 \
--chunk_size 20 \
--total_episode_num 1000 
```

