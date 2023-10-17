FROM nvcr.io/nvidia/pytorch:21.03-py3

RUN apt-get update -y

RUN cd /root/ \ 
&& git clone -b dev https://github.com/your_username/your_repository.git /scobots

WORKDIR /root/scobots

RUN cd /root/scobots/ \ 
&& pip install -r requirements.txt

# TODO: install OC_Atari




