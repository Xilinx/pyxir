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

#include <memory>
#include <unordered_set>

#if defined(USE_VAI_RT_DPUCADX8G) || defined(USE_VAI_RT_DPUCZDX8G)
#include <dpu_runner.hpp>
#include "vai_api/dpu_func.hpp"
#elif defined(USE_VAI_RT_DPUCAHX8H)
#include "xir_api/dpu_func.hpp"
#endif

#include "pyxir/graph/xgraph.hpp"
#include "pyxir/common/xbuffer.hpp"

void vaiDebugMsg(const char *, const char *, const char *, int);
#ifdef DEBUG
#define vaiDebug(x) vaiDebugMsg(x,__FUNCTION__,__FILE__,__LINE__);
#else
#define vaiDebug(x)
#endif

namespace pyxir {
namespace runtime {
namespace vai_rt {

class VaiComputeFunc {

  public:
    VaiComputeFunc(XGraphHolder &xg,
                   const std::string &target,
                   const std::vector<std::string> &in_tensor_names,
                   const std::vector<std::string> &out_tensor_names,
                   const std::string &build_dir);
    ~VaiComputeFunc();

    void operator()(std::vector<XBufferHolder> &in_tensors,
                    std::vector<XBufferHolder> &out_tensors);

    /** @brief Return whether the give operation type is supported */
    bool is_op_supported(const std::string &op_type)
    {
      return supported_ops_.find(op_type) != supported_ops_.end();
    }

  private:
    /** @brief The XGraph */
    XGraphHolder xg_;
    /** @brief The target */
    std::string target_;
    /** @brief The input tensor names in the order that the output buffers will be provided */
    std::vector<std::string> in_tensor_names_;
    /** @brief The input tensor names in the order that the input buffers will be provided */
    std::vector<std::string> out_tensor_names_;
    /** @brief The build directory containing the DPU build files */
    std::string build_dir_;
    /** @brief The connection between outside and internal input tensor order */
    std::vector<int> in_tensor_order_;
    /** @brief The connection between outside and internal output tensor order */
    std::vector<int> out_tensor_order_;
    /** @brief The supported operations by this VAI compute function */
    std::unordered_set<std::string> supported_ops_ =
      {"Input", "Output", "DPUV1", "DPUV2", "DPU", "Tuple", "TupleGetItem", "Transpose"};
    /** @brief In order container for the internal kernel functions */
    std::vector<std::unique_ptr<KernelFunc>> kernel_funcs_;
    /** @brief In order container for the XLayers */
    std::vector<XLayerHolder> Xs_;
    /** @brief The DPU function wrapping Vitis-AI runtime APIs*/
    DpuFunc dpu_func_;
    /** @brief The DPU layer */
    XLayerHolder dpu_X_;

    // VERBOSE
    /** @brief Keep track of total time spent in operator() */
    int64_t total_compute_time_ = 0;
    /** @brief Keep track of kernel timings */
    std::vector<int64_t> total_kernel_times_;
};

} // vai_rt
} // namespace runtime
} // namespace pyxir
