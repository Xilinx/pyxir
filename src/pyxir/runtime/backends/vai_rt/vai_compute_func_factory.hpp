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

#include "pyxir/runtime/compute_func_factory_impl.hpp"

namespace pyxir {
namespace runtime {
namespace vai_rt {

class VaiComputeFuncFactoryImpl : public ComputeFuncFactoryImpl {

  public:
    VaiComputeFuncFactoryImpl(const std::string &runtime)
      : ComputeFuncFactoryImpl(runtime) {}
    virtual ~VaiComputeFuncFactoryImpl() {}

    /**
     * @brief Factory method to create a compute func for the provided XGraph
     *  on the target backend using the Vitis-AI runtime APIs
     */
    ComputeFuncHolder
    get_compute_func(std::shared_ptr<graph::XGraph> &xg,
                     const std::string &target,
                     const std::vector<std::string> &in_tensor_names,
                     const std::vector<std::string> &out_tensor_names,
                     RunOptionsHolder const &run_options = nullptr);
  
};

} // vai_rt
} // namespace runtime
} // namespace pyxir
