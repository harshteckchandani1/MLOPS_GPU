FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y python3 python3-pip && \
    python3 -m pip install --upgrade pip

COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "src/train.py"]


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# # Install dependencies and Python 3.12
# RUN apt update && apt install -y \
#     software-properties-common \
#     curl \
#     gnupg \
#     lsb-release \
#     && curl -fsSL https://packages.python.org/deadsnakes/deadsnakes.asc | tee /etc/apt/trusted.gpg.d/deadsnakes.asc \
#     && add-apt-repository ppa:deadsnakes/ppa \
#     && apt update \
#     && apt install -y python3.12 python3.12-venv python3-pip

# # Set Python 3.12 as the default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# # Install pip for Python 3.12
# RUN python3 -m pip install --upgrade pip

# WORKDIR /app

# COPY . /app

# # Install dependencies from requirements.txt
# RUN pip install -r requirements.txt

# CMD ["python", "src/train.py"]
