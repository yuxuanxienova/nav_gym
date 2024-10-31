FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04 AS builder

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

LABEL maintainer="Filip Bjelonic" mail="filipb@leggedrobotics.com"

# To avoid tzdata asking for geographic location...
ENV LANG C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_frontend noninteractive

# Create folders for cloning
ENV WORKSPACE=/isaac_ws
RUN mkdir -p $WORKSPACE && mkdir -p $WORKSPACE/isaacgym && mkdir -p $WORKSPACE/nav_gym

ENV CXX=/usr/bin/g++
ENV CC=/usr/bin/gcc

# Install dependencies
RUN apt-get update --fix-missing && apt-get install -y apt-utils git wget vim openscad python3-dev python3-distutils python3-pip libpython3-dev
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install matplotlib tensorboard trimesh warp-lang trimesh pytest GitPython numpy==1.23.0 imageio ninja pillow pyyaml scipy
RUN pip3 install opencv-python
RUN pip3 install open3d
RUN pip3 install rtree

# Install IsaacGym
ADD isaacgym $WORKSPACE/isaacgym
RUN cd $WORKSPACE/isaacgym/python && pip install -e .

# Install NavGym
ADD nav_gym $WORKSPACE/nav_gym
RUN cd $WORKSPACE/nav_gym && pip install -e .


# delete folders for mounting your changes on the cluster 
# (uncomment the next line if you have changes in rsl_rl that needs to be mounted on the cluster)
#RUN rm -r $WORKSPACE/rsl_rl/* && rm -rf rsl_rl/{,.[!.],..?}*  # deletes normal files && hidden files
RUN rm -r $WORKSPACE/nav_gym/* && rm -rf nav_gym/{,.[!.],..?}*  # deletes normal files && hidden files

RUN mkdir -p $WORKSPACE/nav_gym/logs
