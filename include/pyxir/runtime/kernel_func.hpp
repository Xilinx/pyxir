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

#include <string>

#include "../graph/xlayer.hpp"
// #include "../opaque_func_registry.hpp"

namespace pyxir {
namespace runtime {

class KernelFunc {

  public:
    KernelFunc() {}
    KernelFunc(XLayerHolder &xl) : xl_(xl) {}
    virtual ~KernelFunc() {}

    virtual void operator()(std::vector<XBufferHolder> &in_tensors,
                            std::vector<XBufferHolder> &out_tensors) {};

  public:
    XLayerHolder xl_;
};

typedef std::unique_ptr<KernelFunc> KernelFuncHolder;

// class OpaqueKernelFunc : protected KernelFunc {

//   public:

//     OpaqueKernelFunc(XLayerHolder &xl, const std::string &of_id) 
//       : xl_(xl), of_id_(of_id)
//     {
//       if (!OpaqueFuncRegistry::Exists(of_id_))
//         throw std::runtime_error("Cannot find OpaqueFunc: " + of_id_
//                                  + " to create OpaqueKernelFunc.");
      
//       OpaqueFunc of_init = OpaqueFuncRegistry::Get(of_id_);

//       of_init(xl_, of_);
//     }

//   void operator()(std::vector<XBufferHolder> &in_tensors,
//                   std::vector<XBufferHolder> &out_tensors)
//   {
//     of_(in_tensors, out_tensors);
//   }

//   protected:
//     std::string of_id_;
//     OpaqueFunc of_;
// }


} // namespace runtime
} // namespace pyxir
