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

#include <unordered_set>
#include <dpu_runner.hpp>

#include "pyxir/pyxir_api.hpp"
#include "pyxir/graph/xlayer.hpp"
#include "pyxir/common/xbuffer.hpp"
#include "pyxir/runtime/kernel_func.hpp"

namespace pyxir {
namespace runtime {
namespace vai_rt {

class DpuFunc : public KernelFunc {

  public:
    DpuFunc() {}
    DpuFunc(XLayerHolder &xl, const std::string &build_dir);

    void operator()(std::vector<XBufferHolder> &in_tensors,
                    std::vector<XBufferHolder> &out_tensors);

  private:
    std::vector<std::string> in_tensor_names_;
    std::vector<std::string> out_tensor_names_;

    // The input tensors and output tensors of the accelerator might be
    //  different than the original input and output tensors
    std::vector<vitis::ai::Tensor*> dpu_runner_in_tensors_;
    std::vector<vitis::ai::Tensor*> dpu_runner_out_tensors_;
    std::vector<int> in_tensor_order_;
    std::vector<int> out_tensor_order_;

    std::unique_ptr<vitis::ai::DpuRunner> dpu_runner_;
};

} // vai_rt
} // namespace runtime
} // namespace pyxir
