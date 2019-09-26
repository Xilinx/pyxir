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
#include <stdexcept>
#include <functional>

#include "compute_func_info.hpp"
#include "../opaque_func.hpp"


namespace pyxir {
namespace runtime {

class IComputeFunc {
  
  public:
    IComputeFunc() {}
    virtual ~IComputeFunc() {}

    virtual void operator()(std::vector<XBufferHolder> &in_tensors,
                            std::vector<XBufferHolder> &out_tensors) = 0;
    
};

class OpaqueComputeFunc : public IComputeFunc {
  
  public:
    OpaqueComputeFunc(OpaqueFuncHolder &of) { of_ = of; }

    virtual void operator()(std::vector<XBufferHolder> &in_tensors,
                            std::vector<XBufferHolder> &out_tensors)
    {
      // Computation is embedded inside OpaqueFunc
      (*of_)(in_tensors, out_tensors);
    }

  private:
    OpaqueFuncHolder of_;
    
};

/**
 * @brief StatefulComputeFunc allows creation of custom compute functions
 *  with state. For this it's necessary to provide a ComputeFuncInfo 
 *  instance which wraps an allocation, compute and destruction function
 */ 
class StatefulComputeFunc : public IComputeFunc {
  
  public:
    StatefulComputeFunc(ComputeFuncInfo &cfi) : cfi_(cfi)
    {
      cfi_.alloc_func(&func_state_);
    }

    virtual void operator()(std::vector<XBufferHolder> &in_tensors,
                            std::vector<XBufferHolder> &out_tensors)
    {
      cfi_.compute_func(func_state_, in_tensors, out_tensors);
    }

    ~StatefulComputeFunc()
    {
      cfi_.release_func(func_state_);
    }

  private:
    ComputeFuncInfo cfi_;
    FuncState func_state_;
};

} // namespace runtime

typedef std::unique_ptr<runtime::IComputeFunc> ComputeFuncHolder;

} // namespace pyxir