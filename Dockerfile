FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        unzip \
        git \
        nginx \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ostris/ai-toolkit.git

COPY dist /root/ai-tookit/
    
COPY nginx/* /etc/nginx/conf.d/

RUN sed -i 's/^user\s\+[^;]\+;/user root;/' /etc/nginx/nginx.conf

RUN pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

COPY extra/* /root/ai-toolkit/

RUN cd ai-toolkit

RUN pip3 install -r requirements.txt && pip3 install -r requirements_extra.txt

COPY configs/* /root/ai-toolkit/config/examples/run/

RUN chmod +x /root/ai-toolkit/start.sh

CMD ["/root/ai-toolkit/start.sh"]
