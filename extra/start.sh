#! /bin/bash
mv /root/ai-toolkit/adamw_fp8.py /root/ai-toolkit/toolkit/optimizers/ && mv /root/ai-toolkit/optimizer.py /root/ai-toolkit/toolkit/
mkdir -p /home/waas/train_data
cd /home/waas/train_data && mkdir -p datasets data output logs
ln -sf /home/waas/train_data/*  /root/ai-toolkit/
nginx -s reload
cd /root/ai-toolkit/ && python3 interface.py >> /tmp/interface_runtime.log 2>&1
