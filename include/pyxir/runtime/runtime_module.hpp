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

#include <vector>

#include "compute_func.hpp"


namespace pyxir {
namespace runtime {

// class IRuntimeModule {

//   public:
//     IRuntimeModule() {}
//     virtual ~IRuntimeModule() {}

//     virtual void execute(std::vector<XBufferHolder> in_tensors,
//                          std::vector<XBufferHolder> out_tensors) = 0;

// };


class RuntimeModule {

  public:
    RuntimeModule() {}
    RuntimeModule(ComputeFuncHolder &compute_func)
    { 
      compute_func_ = std::move(compute_func);
    }

    virtual void execute(std::vector<XBufferHolder> &in_tensors,
                         std::vector<XBufferHolder> &out_tensors)
    {
      (*compute_func_)(in_tensors, out_tensors);
    }

    virtual ~RuntimeModule() {}

  protected:
    ComputeFuncHolder compute_func_ = nullptr;
};
    
} // namespace runtime

// typedef std::unique_ptr<runtime::IRuntimeModule> IRtModHolder;
typedef std::unique_ptr<runtime::RuntimeModule> RtModHolder;

} // namespace pyxir
