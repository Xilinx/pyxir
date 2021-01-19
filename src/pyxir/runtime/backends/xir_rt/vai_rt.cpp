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

/* header file for Vitis-AI Runner APIs */

#include "pyxir/runtime/runtime.hpp"
#include "vai_compute_func_factory.hpp"

namespace pyxir {
namespace runtime {


// Registration of runtime module factory implementations

REGISTER_RUNTIME_FACTORY_IMPL(pyxir::runtime::pxXirRuntimeModule)
  .set_impl(
    new pyxir::runtime::DefaultRuntimeModuleFactoryImpl(
        pyxir::runtime::pxXirRuntimeModule,
        pyxir::runtime::vaiTargets)
  )
  .set_compute_impl(
    new vai_rt::VaiComputeFuncFactoryImpl(pxXirRuntimeModule)
  )
  .set_supported_targets(pyxir::runtime::vaiTargets);

} // namespace runtime
} // namespace pyxir
