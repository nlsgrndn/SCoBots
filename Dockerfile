FROM nvcr.io/nvidia/pytorch:21.03-py3

RUN apt-get update -y
RUN apt-get install libgl1-mesa-glx

RUN cd /root/ \ 
&& git clone -b dev https://github.com/nlsgrndn/SCoBots.git

WORKDIR /root

RUN pip install --upgrade pip

RUN cd /root/SCoBots \
&& pip install -r requirements.txt \
&& pip install ocatari \
&& pip install "gymnasium[atari, accept-rom-license]"




