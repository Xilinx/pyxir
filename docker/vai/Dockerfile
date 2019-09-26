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

FROM xilinx/vitis-ai:latest

RUN apt-get update && apt-get install -y --no-install-recommends\
    build-essential\
    ca-certificates\
    cmake\
    sudo\
    wget\
    git\
    vim\
    graphviz\
    python-dev\
    gnupg2

RUN apt-get update && apt-get install -y gcc-aarch64-linux-gnu

RUN apt-get update && apt-get install -y python-pip python-dev python3.6 python3.6-dev
RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3

COPY install/ubuntu_install_llvm.sh /install/ubuntu_install_llvm.sh
RUN bash /install/ubuntu_install_llvm.sh

COPY install/ubuntu_install_antlr.sh /install/ubuntu_install_antlr.sh
RUN bash /install/ubuntu_install_antlr.sh

COPY install/ubuntu_install_cmake.sh /install/ubuntu_install_cmake.sh
RUN bash /install/ubuntu_install_cmake.sh

RUN . $VAI_ROOT/conda/etc/profile.d/conda.sh && \
    conda activate vitis-ai-tensorflow && \
    pip install --no-cache-dir antlr4-python3-runtime

ARG user
ARG uid
ARG gid
ARG gname
RUN groupadd $gname -g $gid -f && useradd -g $gname -ms /bin/bash $user -u $uid && usermod -aG sudo $user
RUN passwd -d $user

ENV PATH="/opt/vitis_ai/compiler/dnnc/dpuv2:${PATH}"
