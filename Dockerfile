FROM nvcr.io/nvidia/pytorch:21.03-py3

# Install system dependencies
RUN apt-get update -y \
    && apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip

# Clone the project
RUN cd /root/ \ 
&& git clone -b dev https://github.com/nlsgrndn/SCoBots.git

WORKDIR /root

# Install dependencies
RUN cd /root/SCoBots \
&& pip install -r requirements.txt \
&& pip install ocatari \
&& pip install "gymnasium[atari, accept-rom-license]"



