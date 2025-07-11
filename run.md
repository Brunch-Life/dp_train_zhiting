### env creation

```bash
pip install -r requirements.txt
pip install numpy==1.23.5 opencv-python==4.11.0.86 sapien==3.0.0b1 huggingface_hub==0.25.2
```


### dataset convert

```bash
# python datasets/format_convert_delta.py # we not use this script
PYTHONPATH=. python datasets/convert_bingwen.py \
--root_dir "dataset/raw/test_1" \
--save_dir "dataset/processed/tomato_plate_wooden"
```


### train
```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/train.py \
--data_root dataset/processed/tomato_plate_wooden \
--batch_size 128 \
--chunk_size 20 \
--num_steps 200_000 \
--save_every 20_000 \
--lr 1e-5 \
--train_seed 0 \
--episode_num 500 \
--validate_every 2_000
```

