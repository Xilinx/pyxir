# Copyright 2020 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends\
	build-essential\
 	ca-certificates\
    apt-transport-https\
 	cmake\
	sudo\
 	wget\
 	git\
 	vim\
    graphviz\
    python-dev

# python 3.6
# RUN apt-get update 
# RUN apt-get install -y software-properties-common python-software-properties
# RUN add-apt-repository -k hkp://keyserver.ubuntu.com:80 -y ppa:deadsnakes/ppa
# RUN apt-get update && apt-get install -y python-pip python-dev python3.6 python3.6-dev
# RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3

COPY install/ubuntu_install_antlr.sh /install/ubuntu_install_antlr.sh
RUN bash /install/ubuntu_install_antlr.sh

COPY install/ubuntu_install_llvm.sh /install/ubuntu_install_llvm.sh
RUN bash /install/ubuntu_install_llvm.sh

COPY install/ubuntu_install_python.sh /install/ubuntu_install_python.sh
RUN bash /install/ubuntu_install_python.sh

# Install pip
# RUN cd /tmp && wget -q https://bootstrap.pypa.io/get-pip.py && python2 get-pip.py && python3.6 get-pip.py

# RUN pip install --upgrade pip
RUN pip3 install \
    antlr4-python3-runtime \
	tensorflow==1.15 \
    onnx==1.5.0 \
	numpy==1.* \
	pydot==1.4.1 \
	h5py==2.8.0 \
    opencv-python \
	matplotlib \
	jupyter \
    psutil

ARG user
ARG uid
ARG gid
ARG gname
RUN groupadd $gname -g $gid -f && useradd -g $gname -ms /bin/bash $user -u $uid && usermod -aG sudo $user
RUN passwd -d $user
