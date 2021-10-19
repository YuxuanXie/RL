FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64


# COPY sources.list  /etc/apt/sources.list 
# RUN sed -i '/nvidia/d' /etc/apt/sources.list
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget tmux sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*

# Install python3 pip3
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update -y && apt install python3.7 -y && apt-get -y install python3-pip && python3.7 -m pip install --upgrade pip

# # Python packages we use (or used at one point...)
RUN pip3 install mlagents==0.18.0

# # Solving slow pip
# ENV http_proxy http://proxy-chain.xxx.com:911/
# ENV https_proxy http://proxy-chain.xxx.com:912/ 

# #### -------------------------------------------------------------------
# #### install pytorch
# #### -------------------------------------------------------------------
# RUN pip3 install torch==1.4.0+cu92  torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html  -i https://pypi.douban.com/simple/
# RUN pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html --default-timeout=1000 
COPY . .
RUN pip install torch-1.5.0+cu92-cp37-cp37m-linux_x86_64.whl 
RUN rm -rf torch-1.5.0+cu92-cp37-cp37m-linux_x86_64.whl

# -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple ipython
# ## -- Yuxuan 
# RUN pip install tensorboardX -i https://pypi.douban.com/simple/

# WORKDIR /pymarl
