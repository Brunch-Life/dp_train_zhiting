export PATH="/iag_ad_01/ad/tangyinzhou/env/conda/miniconda3/envs/robodiff/bin:$PATH"

export http_proxy=http://proxy-hk.hs.com:3128
export https_proxy=http://proxy-hk.hs.com:3128
export HTTP_PROXY=http://proxy-hk.hs.com:3128
export HTTPS_PROXY=http://proxy-hk.hs.com:3128

apt-get update -y
apt-get install -y libturbojpeg

cd /iag_ad_01/ad/tangyinzhou/tyz/reward_diffusion_policy_relative/diffusion_policy
python train.py \
    --ckpt_dir /iag_ad_01/ad/tangyinzhou/tyz/reward_diffusion_policy_relative/diffusion_policy/data/diffusion_policy_checkpoints_pickplace/ \
    --data_root /iag_ad_01/ad/tangyinzhou/tyz/reward_diffusion_policy_relative/diffusion_policy/data/pickplace \
    --chunk_size 20