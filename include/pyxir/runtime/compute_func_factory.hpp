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

#include "../pyxir_api.hpp"
#include "compute_func.hpp"
#include "run_options.hpp"
#include "compute_func_factory_impl.hpp"


namespace pyxir {
namespace runtime {

class ComputeFuncFactory {
  
  public:

    typedef std::unique_ptr<ComputeFuncFactory> ComputeFuncFactoryHolder;

    ComputeFuncFactory() {}

    PX_API ComputeFuncFactory &set_impl(ComputeFuncFactoryImpl *impl)
    {
      impl_ = ComputeFuncFactoryImplHolder(impl);
      return *this;
    }
    
    PX_API ComputeFuncFactoryImplHolder &get_impl() { return impl_; }

    /**
     * @brief Register a runtime module factor implementation for the given
     *  runtime
     * @param runtime The name of the runtime for which an implementation
     *  is registered
     * @param impl The implementation that will be registered for the given
     *  runtime
     */
    PX_API static ComputeFuncFactory &
    RegisterImpl(const std::string &runtime);
    
    /**
     * @brief Factory method to create a runtime module for the provided XGraph
     *  on the target backend and using the specified runtime
     * @param xg The XGraph model to be executed
     * @param target The target for executing the XGraph model
     * @param in_tensor_names The names of the input tensors that will be
     *  provided in the same order
     * @param out_tensor_names The names of the output tensors that will be
     *  provided in the same order
     * @param runtime The runtime to be used for executing the model
     * @param run_options The specified run options, e.g. whether online 
     *  quantization should be enabled
     * @returns A compute func that can be used for execution of the provided
     *  XGraph
     */
    PX_API static ComputeFuncHolder
    GetComputeFunc(std::shared_ptr<graph::XGraph> &xg,
                    const std::string &target,
                    const std::vector<std::string> &in_tensor_names,
                    const std::vector<std::string> &out_tensor_names,
                    const std::string &runtime,
                    RunOptionsHolder const &run_options = nullptr);

    PX_API static bool Exists(const std::string &runtime);

    class Manager;
  
  private:
    ComputeFuncFactoryImplHolder impl_;
};

typedef std::unique_ptr<ComputeFuncFactory> ComputeFuncFactoryHolder;

} // namespace runtime
} // namespace pyxir