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


namespace pyxir {
namespace runtime {

class ComputeFuncRegistry {
  
  public:

    typedef std::unique_ptr<ComputeFuncRegistry> ComputeFuncRegistryHolder;
    typedef std::function<ComputeFuncHolder ()> FactoryFuncType;

    ComputeFuncRegistry() {}

    PX_API ComputeFuncRegistry &set_factory_func(FactoryFuncType factory_func)
    {
      factory_func_ = factory_func;
      return *this;
    }
    
    PX_API FactoryFuncType &get_factory_func() { return factory_func_; }

    PX_API static bool Exists(const std::string &cf_type);

    /**
     * @brief Register a compute function type with factory method
     * @param cf_type The compute function type
     * @returns A ComputeFuncRegistry instance
     */
    PX_API static ComputeFuncRegistry &Register(const std::string &cf_type);

    /**
     * @brief Register a compute function type with factory method
     * @param cf_type The compute function type
     * @returns The corresponding compute function
     */
    PX_API static ComputeFuncHolder GetComputeFunc(const std::string &cf_type);

    class Manager;
  
  private:
    FactoryFuncType factory_func_;
};

typedef std::unique_ptr<ComputeFuncRegistry> ComputeFuncRegistryHolder;

#define COMPUTE_FUNC_REG_VAR_DEF                   \
  static ::pyxir::runtime::ComputeFuncRegistry&  __mk_ ## PX

#define REGISTER_COMPUTE_FUNC_TYPE(CFTName)\
  STR_CONCAT(COMPUTE_FUNC_REG_VAR_DEF, __COUNTER__) = \
  ComputeFuncRegistry::Register(CFTName)

} // namespace runtime
} // namespace pyxir