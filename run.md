### env creation

```bash
pip install -r requirements.txt
pip install numpy==1.23.5 opencv-python==4.11.0.86 sapien==3.0.0b1 huggingface_hub==0.25.2 diffusers==0.11.1
```


### dataset convert

```bash
python datasets/convert_bingwen.py \
--root_dir "/home/chenyinuo/data/dataset/nonstop/" \
--save_dir "dataset/processed/nonstop_plate_wooden" \
--tasks-dir nonstop_plate_wooden \
--max_task_num 1
```


### train
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
--data_root dataset/processed/nonstop_plate_wooden \
--batch_size 128 \
--lr 1e-5 \
--train_seed 0 \
--num_steps 200_000 \
--save_every 5_000 \
--validate_every 2_000 \
--chunk_size 20 \
--total_episode_num 500 
```


### run in compute sink
```bash 
source /iag_ad_01/ad/tangyinzhou/bingwen/Documents/dp_train_zhiting/start_server.sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train.py \
--data_root dataset/processed/nonstop_plate_wooden_1000 \
--batch_size 128 \
--lr 1e-5 \
--train_seed 0 \
--num_steps 300_000 \
--save_every 20_000 \
--validate_every 2_000 \
--chunk_size 20 \
--wandb_offline \
--total_episode_num 200
```
