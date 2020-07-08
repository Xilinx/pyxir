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
#include "constants.hpp"
#include "runtime_module_factory.hpp"
#include "kernel_func_factory.hpp"

#define STR_CONCAT_(__x, __y) __x##__y
#define STR_CONCAT(__x, __y) STR_CONCAT_(__x, __y)

#define RUNTIME_FACTORY_REG_VAR_DEF                         \
  static ::pyxir::runtime::RuntimeModuleFactory&  __mk_ ## PX

#define REGISTER_RUNTIME_FACTORY_IMPL(RtName)\
  STR_CONCAT(RUNTIME_FACTORY_REG_VAR_DEF, __COUNTER__) = \
  pyxir::runtime::RuntimeModuleFactory::RegisterImpl(RtName)

#define KERNEL_FUNC_REG_VAR_DEF                         \
  static ::pyxir::runtime::KernelFuncFactory&  __mk_ ## PX

#define REGISTER_KERNEL_FUNC(KernelName)\
  STR_CONCAT(KERNEL_FUNC_REG_VAR_DEF, __COUNTER__) = \
  pyxir::runtime::KernelFuncFactory::RegisterImpl(KernelName)
  