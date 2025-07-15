# internet config
export http_proxy="http://proxy.hs.com:3128"
export https_proxy="http://proxy.hs.com:3128"
export HTTP_PROXY="http://proxy.hs.com:3128"
export HTTPS_PROXY="http://proxy.hs.com:3128"

# system config
apt-get update
apt-get install libgl1 -y # for opencv


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/iag_ad_01/ad/tangyinzhou/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/iag_ad_01/ad/tangyinzhou/env/conda/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/iag_ad_01/ad/tangyinzhou/env/conda/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/iag_ad_01/ad/tangyinzhou/env/conda/miniconda3/bin:$PATH"
    fi
fi
# <<< conda initialize <<

conda activate dp_train

# project config
cd /iag_ad_01/ad/tangyinzhou/bingwen/Documents/dp_train_zhiting

