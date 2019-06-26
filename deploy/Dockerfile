#docker build -t fd-detect/python .

FROM python:latest

LABEL maintainer="Yue Dong" \
description="python environment"

COPY requirements /root/

WORKDIR /root/

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  --no-cache-dir -r requirements.txt
