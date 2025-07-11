### dataset convert

```bash
# python datasets/format_convert_delta.py # we not use this script
PYTHONPATH=. python datasets/convert_bingwen.py \
--root_dir "dataset/raw/test_green_bell_pepper/bingwen/data_for_success/" \
--save_dir "dataset/processed/test_green_bell_pepper_delta_bingwen" \
```


### train
```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/train.py \
--data_root dataset/processed/test_green_bell_pepper_delta_bingwen \
--batch_size 128 \
--chunk_size 20 \
--num_steps 200_000 \
--save_every 20_000 \
--lr 1e-5 \
--train_seed 0 \
--episode_num 500 \
--validate_every 2_000
```

