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
#include "common.h"
namespace pyxir {
namespace runtime {
namespace vai_rt {

class DpuFunc : public KernelFunc {

  public:
    DpuFunc() {}
    DpuFunc(XLayerHolder &xl, const std::string &build_dir);
    ~DpuFunc();

    void operator()(std::vector<XBufferHolder> &in_tensors,
                    std::vector<XBufferHolder> &out_tensors);

  private:

    /** @brief The names of the input tensor in the order that they will be provided */
    std::vector<std::string> in_tensor_names_;
    /** @brief The names of the output tensor in the order that they will be provided */
    std::vector<std::string> out_tensor_names_;
    /** @brief The DPU input tensors */
   std::vector<const xir::Tensor*> dpu_runner_in_tensors_;
    /** @brief The DPU output tensors */
   std::vector<const xir::Tensor*> dpu_runner_out_tensors_;
    /** @brief Vector to match the order in which input tensors will be provided with
        the order in which the DPU expects them */
    std::vector<int> in_tensor_order_;
    /** @brief Vector to match the order in which output tensors will be provided with
        the order in which the DPU expects them */
    std::vector<int> out_tensor_order_;
    /** @brief Holder for the DPU runner that will be created using Vitis AI API's */
    //std::unique_ptr<vitis::ai::DpuRunner> dpu_runner_;
    std::unique_ptr<xir::Graph> graph;
    std::vector<const xir::Subgraph*> subgraph;
    std::unique_ptr<vart::Runner> runner;

    // VERBOSE
    /** @brief The total time spent in async DPU call */
    int64_t total_async_time_ = 0;
    /** @brief The total time spent in wait DPU call */
    int64_t total_wait_time_ = 0;
    /** @brief The total time spent in operator() call */
    int64_t total_dpu_time_ = 0;
};

} // vai_rt
} // namespace runtime
} // namespace pyxir
