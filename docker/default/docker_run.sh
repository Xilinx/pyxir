#!/usr/bin/env bash

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

# This file should be executed from pyxir/docker directory 
#   to keep the context as small as possible

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

user=`whoami`
docker=`dirname $SCRIPT_DIR`
pyxir=`dirname $docker`
workspace=`pwd`

#sudo \
docker run \
  --rm \
  --net=host \
  -it \
  --privileged=true \
  --user $user \
  -v /home/$user:/home/$user \
  -v /dev:/dev \
  -v $workspace:/workspace \
  -v $SCRIPT_DIR:/docker \
  -w /workspace \
  $user/pyxir \
  bash --login /docker/setup_after_container_startup.sh

