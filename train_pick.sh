export PATH="/iag_ad_01/ad/tangyinzhou/env/conda/miniconda3/envs/robodiff/bin:$PATH"

cd /iag_ad_01/ad/tangyinzhou/tyz/reward_diffusion_policy_relative/diffusion_policy
python train.py \
    --ckpt_dir /iag_ad_01/ad/tangyinzhou/tyz/reward_diffusion_policy_relative/diffusion_policy/data/diffusion_policy_checkpoints_pick/ \
    --data_root /iag_ad_01/ad/tangyinzhou/tyz/reward_diffusion_policy_relative/diffusion_policy/data/pick \
    --chunk_size 20