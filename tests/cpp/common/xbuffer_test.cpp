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
#include <vector>

#include <catch2/catch.hpp>

#include "pyxir/common/xbuffer.hpp"
#include "pyxir/opaque_func_registry.hpp"


TEST_CASE("Test XBuffer initialization")
{
  std::array<float, 16> zeros = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // Main constructor
  std::array<float, 16> x = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  pyxir::XBuffer xb = pyxir::XBuffer((void *) &x[0], 4, "f", 4,
                                     std::vector<ssize_t>{1, 1, 4, 4},
                                     false, false);
  std::array<float, 16> y;
  std::copy((float *) xb.data, (float *) xb.data + 16, std::begin(y));
  REQUIRE(!xb.own_data);
  REQUIRE(xb.itemsize == 4);
  REQUIRE(xb.format == "f");
  REQUIRE(xb.ndim == 4);
  REQUIRE(xb.shape == std::vector<ssize_t>{1, 1, 4, 4});
  REQUIRE(xb.size == 16);
  REQUIRE(x == y);

  // Copy constructor
  pyxir::XBuffer xb2(xb);
  float *data2 = (float *) xb2.data;
  data2[0] = 0.0f;
  std::array<float, 16> y2;
  std::copy((float *) xb2.data, (float *) xb2.data + 16, std::begin(y2));
  std::array<float, 16> y2_expected = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  REQUIRE(y2 == y2_expected);
  REQUIRE(x == y);

  // Move constructor
  std::vector<pyxir::XBuffer> data;
  data.push_back(pyxir::XBuffer((void *) &x[0], 4, "f", 4,
                                 std::vector<ssize_t>{1, 1, 4, 4},
                                 false, false));
  std::array<float, 16> y3;
  std::copy((float *) data[0].data, (float *) data[0].data + 16, std::begin(y3));
  REQUIRE(x == y3);

  // Copy assignment with deallocation of passed heap memory
  float *x4 =  new float[16];
  for (size_t i = 0; i < 16; ++i) x4[i] = 0.0f;
  pyxir::XBuffer xb4 = pyxir::XBuffer((void *) x4, 4, "f", 4,
                                      std::vector<ssize_t>{1, 1, 4, 4},
                                      false, true);
  xb4 =  xb;
  std::array<float, 16> y4;
  std::copy((float *) xb4.data, (float *) xb4.data + 16, std::begin(y4));
  REQUIRE(xb4.itemsize == 4);
  REQUIRE(xb4.format == "f");
  REQUIRE(xb4.ndim == 4);
  REQUIRE(xb4.shape == std::vector<ssize_t>{1, 1, 4, 4});
  REQUIRE(xb4.size == 16);
  REQUIRE(xb4.own_data);
  REQUIRE(x == y4);
  float *data4 = (float *) xb4.data;
  data4[0] = 0.0f;
  std::copy((float *) xb4.data, (float *) xb4.data + 16, std::begin(y4));

  // Move assignment with data ownership transfership
  float *x5 =  new float[16];
  for (size_t i = 0; i < 16; ++i) x5[i] = 0.0f;
  pyxir::XBuffer xb5 = pyxir::XBuffer((void *) x5, 4, "f", 4,
                                      std::vector<ssize_t>{1, 1, 4, 4},
                                      false, true);
  pyxir::XBuffer xb5_final = pyxir::XBuffer((void *) &x[0], 4, "f", 4,
                                            std::vector<ssize_t>{1, 1, 4, 4},
                                            false, false);
  REQUIRE(xb5.own_data);
  REQUIRE(!xb5_final.own_data);
  xb5_final = std::move(xb5);
  REQUIRE(!xb5.own_data);
  REQUIRE(xb5_final.own_data);
  std::array<float, 16> y5;
  std::copy((float *) xb5_final.data, (float *) xb5_final.data + 16, std::begin(y5));
  REQUIRE(y5 == zeros);
}

TEST_CASE("Test XBuffer FFI")
{
  std::array<float, 36> x = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0};
  std::array<float, 36> y = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0};
  
  pyxir::XBufferHolder xb_in = std::shared_ptr<pyxir::XBuffer>(
    new pyxir::XBuffer((void *) &x[0], 4, "f", 4,
                       std::vector<ssize_t>{2, 2, 3, 3}, false, false));
  pyxir::XBufferHolder xb_out = std::shared_ptr<pyxir::XBuffer>(
    new pyxir::XBuffer((void *) &y[0], 4, "f", 4,
                       std::vector<ssize_t>{2, 2, 3, 3}, false, false));

  std::vector<pyxir::XBufferHolder> in_buffers{xb_in};
  std::vector<pyxir::XBufferHolder> out_buffers{xb_out};

  REQUIRE(pyxir::OpaqueFuncRegistry::Exists("pyxir.test.copy_xbuffers"));
  pyxir::OpaqueFunc of = pyxir::OpaqueFuncRegistry::Get("pyxir.test.copy_xbuffers");
  of(in_buffers, out_buffers);
  REQUIRE(y == x);
}


