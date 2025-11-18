FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y python3 python3-pip

COPY . /app

RUN pip3 install -r requirements.txt

CMD ["python3", "src/train.py"]
