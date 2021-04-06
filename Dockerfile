FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04
FROM python:3.6
FROM ros:kinetic-ros-base-xenial

ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt-get install -y python3.6


RUN apt-get update && apt-get upgrade -y && apt-get install -y \
        git \
        build-essential \
        cmake \
        vim \
        wget \
        pkg-config \
        #libjpeg62-turbo-dev \
        libtiff5-dev \
        #libjasper-dev \
        #libpng12-dev \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libatlas-base-dev \
        gfortran \
        libc6-dev-i386 \
        libavresample-dev \
        libgphoto2-dev \
        #libx32gcc-4.8-dev \
        libgstreamer-plugins-base1.0-dev \
        libdc1394-22-dev \
        lsb-release \
        gnupg2 \
        curl \
        nano \
        python3-pip \
        wget

        
# Update pip
RUN /bin/bash -c "python3.6 -m pip install pip --upgrade"
RUN /bin/bash -c "python3.6 -m pip install wheel"


RUN pip install --upgrade setuptools

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
RUN mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
RUN apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda

# Install ROS Kinetic
#RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update
RUN apt-get install -y ros-kinetic-desktop-full
RUN apt install -y --no-install-recommends python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential python-catkin-tools python-catkin-pkg ros-kinetic-joy* ros-kinetic-controller-manager-msgs 

#RUN rosdep init && rosdep update
   

COPY . /python3_ws/src/

WORKDIR /python3_ws/src/

RUN pip install -r requirements.txt
WORKDIR /python3_ws/
RUN chmod +x -R src/
RUN /bin/bash -c "source /opt/ros/kinetic/setup.bash"
RUN echo "source /opt/ros/kinetic/setup.bash" >> /root/.bashrc
RUN /bin/bash -c "source /root/.bashrc"
ENV PYTHONPATH=$PYTHONPATH:/usr/bin/python3.6
RUN . /opt/ros/kinetic/setup.sh && catkin_make -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3.6
RUN /bin/bash -c "source /python3_ws/devel/setup.bash"
# Set Python3.6 as Python version by default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
RUN apt-get install -y x11vnc xvfb 
RUN mkdir ~/.vnc
RUN x11vnc -storepasswd 1234 ~/.vnc/passwd


# Expose Tensorboard
EXPOSE 6006
EXPOSE 5900


