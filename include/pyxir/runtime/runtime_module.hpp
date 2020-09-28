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
#include "../common/serializable.hpp"
#include "../runtime/compute_func_registry.hpp"


namespace pyxir {
namespace runtime {

// class IRuntimeModule {

//   public:
//     IRuntimeModule() {}
//     virtual ~IRuntimeModule() {}

//     virtual void execute(std::vector<XBufferHolder> in_tensors,
//                          std::vector<XBufferHolder> out_tensors) = 0;

// };


class RuntimeModule : public ISerializable {

  public:
    RuntimeModule() {}
    RuntimeModule(ComputeFuncHolder &compute_func,
                  const std::vector<std::string> &in_tensor_names,
                  const std::vector<std::string> &out_tensor_names)
                  : in_tensor_names_(in_tensor_names),
                    out_tensor_names_(out_tensor_names)
    { 
      compute_func_ = std::move(compute_func);
    }

    virtual void execute(std::vector<XBufferHolder> &in_tensors,
                         std::vector<XBufferHolder> &out_tensors)
    {
      (*compute_func_)(in_tensors, out_tensors);
    }

    std::vector<std::string> get_in_tensor_names() { return in_tensor_names_; }

    std::vector<std::string> get_out_tensor_names() { return out_tensor_names_; }

    virtual void serialize_px(PxOStringStream &pstream)
    {
      pstream.write(compute_func_->get_type());
      compute_func_->serialize_px(pstream);
      
      // Serialize in and out tensor names
      pstream.write(in_tensor_names_.size());
      for (auto & it : in_tensor_names_) {
        pstream.write(it);
      }
      pstream.write(out_tensor_names_.size());
      for (auto & ot : out_tensor_names_) {
        pstream.write(ot);
      }
    }

    virtual void deserialize_px(PxIStringStream &pstream)
    {
      std::string cf_type;
      pstream.read(cf_type);
      compute_func_ = ComputeFuncRegistry::GetComputeFunc(cf_type);
      compute_func_->deserialize_px(pstream);

      // Deserialize in and out tensor names
      int it_size;
      pstream.read(it_size);
      for (int i = 0; i < it_size; ++i) {
        std::string it_name;
        pstream.read(it_name);
        in_tensor_names_.push_back(it_name);
      }
      int ot_size;
      pstream.read(ot_size);
      for (int i = 0; i < ot_size; ++i) {
        std::string ot_name;
        pstream.read(ot_name);
        out_tensor_names_.push_back(ot_name);
      }
    }

    virtual ~RuntimeModule() {}

  protected:
    ComputeFuncHolder compute_func_ = nullptr;
    std::vector<std::string> in_tensor_names_;
    std::vector<std::string> out_tensor_names_;
};
    
} // namespace runtime

typedef std::unique_ptr<runtime::RuntimeModule> RtModHolder;

} // namespace pyxir
