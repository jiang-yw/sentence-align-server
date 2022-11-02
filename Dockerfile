FROM python:3.7.15-slim

COPY ./app /app

RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g; s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list \
    && pip install -i https://pypi.douban.com/simple/ -U pip \
    && pip config set global.index-url https://pypi.douban.com/simple/ \
    && apt-get update -y && apt-get upgrade -y \
    && pip install -r /app/requirements.txt \
    && python -m laserembeddings download-models \
    && rm  -rf /var/lib/apt/lists/*

WORKDIR /app

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8086"]
