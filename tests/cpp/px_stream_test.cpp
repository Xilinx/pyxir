/*
 *  Copyright 2020 Xilinx Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *  
 *      http://www.apache.org/licenses/LICENSE-2.0
 *  
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <iostream>
#include <memory>

#include <catch2/catch.hpp>

#include "pyxir/common/px_stream.hpp"


TEST_CASE("Test PxStream")
{
  std::ostringstream sstream;
  pyxir::PxOStringStream opxs(sstream);

  opxs.write("Example");
  assert(sstream.str() == " 7 Example");

  opxs.write(false);
  opxs.write(1);
  assert(sstream.str() == " 7 Example 1 0 1 1");
}

TEST_CASE("Test iPxStream")
{
  std::istringstream sstream(" 7 Example 1 0 1 1");
  pyxir::PxIStringStream ipxs(sstream);

  std::string str;
  ipxs.read(str);
  assert(str == "Example");

  bool b;
  ipxs.read(b);
  assert(b == false);

  int i;
  ipxs.read(i);
  assert(i == 1);
}