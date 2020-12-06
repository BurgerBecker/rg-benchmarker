# Use Ubuntu 16.04 as base image
FROM ubuntu:

git clone https://github.com/BurgerBecker/rg-benchmarker.git

WORKDIR rg-benchmarker/
ADD requirements.txt /

RUN \
     apt-get update \
  && apt-get install -y --no-install-recommends \
     build-essential \
     pkg-config \
     python3-dev \
     python-numpy \
     python3-pip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


# Install basic pip tools
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir setuptools==34.3.2 wheel==0.30.0.a0


RUN pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED 0

ENTRYPOINT ["/bin/sh", "-c"]


