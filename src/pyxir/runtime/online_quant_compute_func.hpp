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

#pragma once

#include "pyxir/opaque_func_registry.hpp"
#include "pyxir/graph/xgraph.hpp"
#include "pyxir/common/xbuffer.hpp"
#include "pyxir/runtime/compute_func.hpp"


namespace pyxir {
namespace runtime {

class OnlineQuantComputeFunc {

  public:
    OnlineQuantComputeFunc(XGraphHolder &xg,
                           const std::string &target,
                           const std::vector<std::string> &in_tensor_names,
                           const std::vector<std::string> &out_tensor_names,
                           const std::string &runtime,
                           int nb_quant_inputs);

    void operator()(std::vector<XBufferHolder> &in_tensors,
                    std::vector<XBufferHolder> &out_tensors);

  private:
    XGraphHolder xg_;
    std::string target_;
    std::string runtime_;
    std::vector<std::string> in_tensor_names_;
    std::vector<std::string> out_tensor_names_;
    int nb_quant_inputs_; 
    int count_ = 0;

    ComputeFuncHolder cf_ = nullptr;
    OpaqueFuncHolder quant_of_;
};

} // namespace runtime
} // namespace pyxir
