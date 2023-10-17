FROM nvcr.io/nvidia/pytorch:21.03-py3

# Install system dependencies
RUN apt-get update -y \
    && apt-get install -y libgl1-mesa-glx

# Create a non-root user
RUN useradd -m dockeruser
USER dockeruser

# Clone the repository and set the working directory
WORKDIR /home/dockeruser
RUN git clone -b dev https://github.com/nlsgrndn/SCoBots.git

# Install or upgrade pip and packages
RUN pip install --upgrade pip
RUN pip install -r SCoBots/requirements.txt
RUN pip install ocatari
RUN pip install "gymnasium[atari, accept-rom-license]"

# Set the working directory to the project directory
WORKDIR /home/dockeruser/SCoBots




