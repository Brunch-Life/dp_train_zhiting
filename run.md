### dataset convert

```bash
# python datasets/format_convert_delta.py # we not use this script
PYTHONPATH=. python datasets/convert_bingwen.py
```


### train
```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
--data_root /home/chenyinuo/data/dp/diffusion_policy/data/test_green_bell_pepper_delta_bingwen_new_T \
--batch_size 128 \
--chunk_size 20 \
--num_steps 200_000 \
--save_every 20_000 \
--lr 1e-5 \
--train_seed 0 \
--episode_num 500 \
--validate_every 2_000 \
```

