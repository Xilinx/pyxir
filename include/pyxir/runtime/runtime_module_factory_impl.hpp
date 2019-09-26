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

#include "constants.hpp"
#include "run_options.hpp"
#include "runtime_module.hpp"


namespace pyxir {
namespace runtime {

/**
 * @brief
 */
class RuntimeModuleFactoryImpl {

  public:

    typedef std::function<void(std::vector<XBufferHolder> &,
                               std::vector<XBufferHolder> &)> FuncType_;

    RuntimeModuleFactoryImpl(const std::string &runtime) : rt_name_(runtime) {}
    virtual ~RuntimeModuleFactoryImpl() {}

    /**
     * @brief Factory method to create a runtime module for the provided XGraph
     *  on the target backend and using the specified runtime
     * @param xg The XGraph model to be executed
     * @param target The target for executing the XGraph model
     * @param in_tensor_names The names of the input tensors that will be
     *  provided in the same order
     * @param out_tensor_names The names of the output tensors that will be
     *  provided in the same order
     * @param run_options The specified run options, e.g. whether online 
     *  quantization should be enabled
     * @returns A runtime module that can be used for execution of the provided
     *  XGraph
     */
    virtual RtModHolder 
    get_runtime_module(std::shared_ptr<graph::XGraph> &xg,
                       const std::string &target,
                       const std::vector<std::string> &in_tensor_names,
                       const std::vector<std::string> &out_tensor_names,
                       RunOptionsHolder const &run_options = nullptr) = 0;

  protected:
    std::string rt_name_;
    
};


/**
 * @brief
 */
class DefaultRuntimeModuleFactoryImpl : public RuntimeModuleFactoryImpl {

  public:

    using RuntimeModuleFactoryImpl::FuncType_;

    DefaultRuntimeModuleFactoryImpl(const std::string &runtime)
      : RuntimeModuleFactoryImpl(runtime) {}

    /**
     * @brief Factory method to create a runtime module for the provided XGraph
     *  on the target backend and using the specified runtime
     * @param xg The XGraph model to be executed
     * @param target The target for executing the XGraph model
     * @param in_tensor_names The names of the input tensors that will be
     *  provided in the same order
     * @param out_tensor_names The names of the output tensors that will be
     *  provided in the same order
     * @param run_options The specified run options, e.g. whether online 
     *  quantization should be enabled
     * @returns A runtime module that can be used for execution of the provided
     *  XGraph
     */
    RtModHolder 
    get_runtime_module(std::shared_ptr<graph::XGraph> &xg,
                       const std::string &target,
                       const std::vector<std::string> &in_tensor_names,
                       const std::vector<std::string> &out_tensor_names,
                       RunOptionsHolder const &run_options = nullptr);
    
};

typedef std::unique_ptr<RuntimeModuleFactoryImpl>
  RuntimeModuleFactoryImplHolder;

} // namespace runtime
} // namespace pyxir