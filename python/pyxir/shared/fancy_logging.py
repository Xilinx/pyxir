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

""" Wrapper around Python logging module for some fancy logging features """

import logging


def getLogger(name):
    logger = logging.getLogger(name)
    return FancyLogger(logger)


class FancyLogger(object):

    def __init__(self, logger):
        self.logger = logger

    def banner(self, message):
        # logger = logging.getLogger(logger)

        banner = \
            "\n**************************************************\n" +\
            "* " + message + "\n" +\
            "**************************************************"

        self.logger.info(banner)
