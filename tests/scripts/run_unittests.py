# Copyright 2021 Xilinx Inc.
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

import argparse
import unittest

def discover_and_run(test_dir: str = '.'):
    """Discover and run tests cases, returning the result."""
    loader = unittest.TestLoader()
    tests = loader.discover(test_dir)
    testRunner = unittest.runner.TextTestRunner()
    result = testRunner.run(tests)
    return result


def main(path_to_tests):
    res = discover_and_run(path_to_tests)
    return not res.errors and not res.failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", help="Path to tests")
    args = parser.parse_args()
    path_to_tests = args.D
    assert main(path_to_tests)
