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
#include <unordered_map>
#include <unordered_set>

#include "../pyxir_api.hpp"
#include "constants.hpp"
#include "run_options.hpp"
#include "runtime_module.hpp"
#include "runtime_module_factory_impl.hpp"
#include "compute_func_factory.hpp"

namespace pyxir {
namespace runtime {

class RuntimeModuleFactory {
  
  public:

    typedef std::unique_ptr<RuntimeModuleFactory> RuntimeModuleFactoryHolder;

    RuntimeModuleFactory(const std::string &runtime) : runtime_(runtime) {}

    PX_API RuntimeModuleFactory &set_impl(RuntimeModuleFactoryImplHolder &impl)
    {
      impl_ = std::move(impl);
      return *this;
    }

    PX_API RuntimeModuleFactory &set_impl(RuntimeModuleFactoryImplHolder impl)
    {
      impl_ = std::move(impl);
      return *this;
    }

    PX_API RuntimeModuleFactory &set_impl(RuntimeModuleFactoryImpl *impl)
    {
      impl_ = RuntimeModuleFactoryImplHolder(impl);
      return *this;
    }

    PX_API RuntimeModuleFactoryImplHolder &get_impl() { return impl_; }

    PX_API RuntimeModuleFactory &set_compute_impl(
      ComputeFuncFactoryImpl *compute_impl)
    {
      ComputeFuncFactory::RegisterImpl(runtime_)
        .set_impl(compute_impl);
      return *this;
    }

    PX_API RuntimeModuleFactory &set_supported_targets(const std::vector<std::string> &targets)
    {
      std::copy(targets.begin(), targets.end(),
                std::inserter(supported_targets_, supported_targets_.end()));
      return *this;
    }

    PX_API std::unordered_set<std::string> &get_supported_targets() { return supported_targets_; }

    PX_API bool is_target_supported(const std::string &target)
    { 
      return supported_targets_.find(target) != supported_targets_.end();
    }

    /**
     * @brief Register a runtime module factor implementation for the given
     *  runtime
     * @param runtime The name of the runtime for which an implementation
     *  is registered
     * @param impl The implementation that will be registered for the given
     *  runtime
     */
    PX_API static RuntimeModuleFactory &
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
     * @returns A runtime module that can be used for execution of the provided
     *  XGraph
     */
    PX_API static RtModHolder 
    GetRuntimeModule(std::shared_ptr<graph::XGraph> &xg,
                    const std::string &target,
                    const std::vector<std::string> &in_tensor_names,
                    const std::vector<std::string> &out_tensor_names,
                    const std::string &runtime,
                    RunOptionsHolder const &run_options = nullptr);

    /**
     * @brief Check whether the provided runtime exists
     */
    PX_API static bool Exists(const std::string &runtime);

    /**
     * @brief Check whether the provided runtime supports the provided target
     */
    PX_API static bool SupportsTarget(const std::string &runtime,
                                      const std::string &target);

    class Manager;
  
  private:
    /** @brief The runtime identifier */
    std::string runtime_;
    /** @brief The runtime factory implementation for the stored runtime */
    RuntimeModuleFactoryImplHolder impl_;
    /** @brief The supported targets for the stored runtime */
    std::unordered_set<std::string> supported_targets_;
};

typedef std::unique_ptr<RuntimeModuleFactory> RuntimeModuleFactoryHolder;

} // namespace runtime
} // namespace pyxir