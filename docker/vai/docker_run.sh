#!/bin/bash

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

# HERE=`dirname $(readlink -f $0)` # Absolute path of current directory
# HERE=`pwd -P`

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

user=`whoami`
uid=`id -u`
gid=`id -g`
DOCKER_DIR=`dirname $SCRIPT_DIR`
PYXIR_ROOT=`dirname $DOCKER_DIR`

echo $PYXIR_ROOT

xclmgmt_driver="$(find /dev -name xclmgmt\*)"
docker_devices=""
for i in ${xclmgmt_driver} ;
do
  docker_devices+="--device=$i "
done

render_driver="$(find /dev/dri -name renderD\*)"
for i in ${render_driver} ;
do
  docker_devices+="--device=$i "
done

##############################

if [[ $IMAGE_NAME == *"gpu"* ]]; then
  docker run \
    $docker_devices \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -v $SCRIPT_DIR:/docker \
    -v $PWD:/workspace \
    -w /workspace \
    -it \
    --rm \
    --runtime=nvidia \
    --network=host \
    $user/pyxir/vai \
    bash --login /docker/setup_after_container_startup.sh
else
  docker run \
    $docker_devices \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -e USER=$user -e UID=$uid -e GID=$gid \
    -v $SCRIPT_DIR:/docker \
    -v $PWD:/workspace \
    -w /workspace \
    -${DETACHED}it \
    --rm \
    --network=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    $USER/pyxir/vai \
    bash --login /docker/setup_after_container_startup.sh
fi
