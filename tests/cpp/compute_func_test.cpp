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

#include "pyxir/runtime/compute_func.hpp"

TEST_CASE("Test StatefulComputeFunc")
{
  class Test {

  };

  pyxir::runtime::ComputeFuncInfo cfi;
  cfi.alloc_func = [](pyxir::runtime::FuncState *state) {
    auto* p = new Test();
    *state = (void*) p;
    return 0;
  };

  cfi.compute_func = [](pyxir::runtime::FuncState state, 
                        std::vector<pyxir::XBufferHolder> &in_tensors,
                        std::vector<pyxir::XBufferHolder> &out_tensors)
  {
    Test* t = reinterpret_cast<Test*>(state);
    float* fdata = reinterpret_cast<float*>(out_tensors[0]->data);
    fdata[0] = 5.0;
  };

  cfi.release_func = [](pyxir::runtime::FuncState state) {
    if(state)
      delete reinterpret_cast<Test*>(state);
  };

  pyxir::runtime::StatefulComputeFunc scf(cfi);

  std::array<float, 16> in_arr = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  pyxir::XBufferHolder xb_in = std::shared_ptr<pyxir::XBuffer>(
    new pyxir::XBuffer((void *) &in_arr[0], 4, "f", 4,
                       std::vector<ssize_t>{1, 1, 4, 4}, false, false));

  std::array<float, 8> out_arr = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  pyxir::XBufferHolder xb_out = std::shared_ptr<pyxir::XBuffer>(
    new pyxir::XBuffer((void *) &out_arr[0], 4, "f", 4,
                       std::vector<ssize_t>{1, 2, 2, 2}, false, false));
    
  // for (int i = 0; i < 8; ++i)
  //   std::cout << out_arr[i] << ", " << std::endl;

  std::vector<pyxir::XBufferHolder> in_tensors{xb_in};
  std::vector<pyxir::XBufferHolder> out_tensors{xb_out};

  scf(in_tensors, out_tensors);

  // std::array<float, 8> expected_out = {5.0, 4.0, 4.0, 3.25, 1.0, 1.0, 0.0, 0.25};
  std::array<float, 8> expected_out = {5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  REQUIRE(out_tensors.size() == 1);
  REQUIRE(out_arr == expected_out);

}
