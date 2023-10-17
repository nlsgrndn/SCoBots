#FROM nvcr.io/nvidia/pytorch:21.03-py3
FROM python:3.8

RUN apt-get update -y && apt-get install -y git

RUN cd /root/ \ 
&& git clone -b dev https://github.com/nlsgrndn/SCoBots.git

WORKDIR /root

RUN cd /root/SCoBots \
&& pip install --upgrade pip \
&& pip install -r requirements.txt

# TODO: install OC_Atari




