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

#include <functional>

#include "../pyxir_api.hpp"
#include "kernel_func.hpp"


namespace pyxir {
namespace runtime {

class KernelFuncFactory {
  
  public:

    typedef std::unique_ptr<KernelFuncFactory> KernelFuncFactoryHolder;
    typedef std::function<void (XLayerHolder &, KernelFuncHolder &)> FuncCreatorType;
    //typedef void (*)(XLayerHolder &, KernelFuncHolder &) FuncCreatorType;
    
    KernelFuncFactory() {}

    /**
     * @brief Set a kernel func implementation
     * @param impl The lambda for creating the kernel function
     */
    PX_API KernelFuncFactory &set_impl(void (*impl)(XLayerHolder &, KernelFuncHolder &))
    {
      // impl_ = *static_cast<FuncCreatorType*>(impl);
      impl_ = static_cast<FuncCreatorType>(impl);
      //impl_ = impl;
      return *this;
    }
    
    PX_API FuncCreatorType &get_impl() { return impl_; }

    /**
     * @brief Register a kernel func for the given kernel identifier
     * @param kernel_id The kernel identifier
     */
    PX_API static KernelFuncFactory &
    RegisterImpl(const std::string &kernel_id);
    
    /**
     * @brief Retrieve the kernel func corresponding to the given identifier
     * @param xl The XLayer to initialize the kernel function
     * @returns The kernel func corresponding to the given identifier
     */
    PX_API static KernelFuncHolder
    GetKernelFunc(const std::string &kernel_id, XLayerHolder &xl);

    PX_API static bool Exists(const std::string &kernel_id);

    class Manager;
  
  private:
    FuncCreatorType impl_;
};

typedef std::unique_ptr<KernelFuncFactory> KernelFuncFactoryHolder;

} // namespace runtime
} // namespace pyxir