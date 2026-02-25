# Flower-Trainer: a simple fine-tuning trainer for diffusion models

![](assets/1.png)

## Install

To install flower-trainer, install the package in a [**Python==3.12**](https://www.python.org/) environment with `pip`.

```
apt-get update 
apt-get install -y nginx
cp nginx/* /etc/nginx/conf.d/
sed -i 's/^user\s\+[^;]\+;/user root;/' /etc/nginx/nginx.conf
pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip3 install -r requirements.txt && pip3 install -r requirements_extra.txt
```

## Run

To run flower-trainer, make sure the relevant folders are created.

```
mkdir -p /root/autodl-tmp/train_data
cd /root/autodl-tmp/train_data && mkdir -p datasets data output logs
ln -sf /root/autodl-tmp/train_data/* /root/ai-toolkit/
cd /root/ai-toolkit/ && python3 interface.py >> /tmp/interface_runtime.log 2>&10
nginx &
nginx -s reload
```
