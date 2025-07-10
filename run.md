### dataset convert

```bash
python datasets/format_convert_delta.py 
PYTHONPATH=. python datasets/convert_bingwen.py
```


### train
```bash
CUDA_VISIBLE_DEVICES=6 python train.py \
--data_root /home/chenyinuo/data/reward_diffusion_policy_relative/diffusion_policy/data/test_green_bell_pepper_delta \
--batch_size 128 \
--chunk_size 20 \
--num_steps 200_000 \
--lr 1e-5 \
--seed 0 
```

