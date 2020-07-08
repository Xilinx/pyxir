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

#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <chrono>

#include "pyxir/common/util.hpp"
#include "dpu_func.hpp"

namespace pyxir {
namespace runtime {
namespace vai_rt {

DpuFunc::DpuFunc(XLayerHolder &xl) : KernelFunc(xl)
{
  std::vector<std::string> dpu_in_tensor_names = xl->bottoms;
  std::vector<std::string> dpu_internal_in_tensor_names
    = xl->get_attr("input_names").get_strings();
  
  in_tensor_names_ = dpu_internal_in_tensor_names;

  std::vector<std::string> dpu_out_tensor_names = xl->tops;
  out_tensor_names_ = dpu_out_tensor_names;
  
  std::unordered_map<std::string, std::string> rt_in_map = 
    xl->get_attr("rt_in_map").get_map_str_str();
  std::unordered_map<std::string, std::string> rt_out_map = 
    xl->get_attr("rt_out_map").get_map_str_str();
  // assert(in_tensor_names_.size() == rt_in_map.size());
  // assert(out_tensor_names_.size() == rt_out_map.size());
  
  pxDebug("Before DpuRunner init");
  // Setup DPU runner
  // If PX_BUILD_DIR environment variable is set, we use that directory
  //   to setup the DpuRunner
  std::string model_path;
  const char *env_build_dir = std::getenv("PX_BUILD_DIR");
  if (env_build_dir != NULL) {
    model_path = env_build_dir;
  } else {
    model_path = xl->get_attr("work_dir").get_string();
  }

  auto dpu_runners = vitis::ai::DpuRunner::create_dpu_runner(model_path);
  dpu_runner_ = std::move(dpu_runners[0]);

  if(dpu_runner_->get_tensor_format() != vitis::ai::DpuRunner::TensorFormat::NCHW) {
    pxDebug("Invalid tensor format NHWC");
  }
  pxDebug("After DpuRunner init");
  
  dpu_runner_in_tensors_ = dpu_runner_->get_input_tensors();
  dpu_runner_out_tensors_ = dpu_runner_->get_output_tensors();
  assert(dpu_runner_in_tensors_.size() == dpu_in_tensor_names.size());
  assert(dpu_runner_out_tensors_.size() == dpu_out_tensor_names.size());

  std::vector<std::string> dpu_runner_in_tensor_names;
  std::transform(dpu_runner_in_tensors_.begin(), dpu_runner_in_tensors_.end(),
   std::back_inserter(dpu_runner_in_tensor_names),
   [](vitis::ai::Tensor* t) -> const std::string & { return t->get_name(); });

  std::vector<std::string> dpu_runner_out_tensor_names;
  std::transform(dpu_runner_out_tensors_.begin(), dpu_runner_out_tensors_.end(),
   std::back_inserter(dpu_runner_out_tensor_names),
   [](vitis::ai::Tensor* t) -> const std::string & { return t->get_name(); });

  std::vector<std::string> rt_in_names;
  std::transform(in_tensor_names_.begin(), in_tensor_names_.end(),
   std::back_inserter(rt_in_names),
   [&rt_in_map](const std::string &elem)
   -> const std::string & { return rt_in_map[elem]; });

  for (int i = 0; i < in_tensor_names_.size(); ++i)
  {
    std::string dpu_in_name = dpu_runner_in_tensor_names[i];

    std::vector<std::string>::iterator iter = std::find_if(
      rt_in_names.begin(), rt_in_names.end(),
      [dpu_in_name](const std::string &elem) { return elem == dpu_in_name; });
    size_t index = std::distance(rt_in_names.begin(), iter);
    if (index == rt_in_names.size())
    {
      throw std::runtime_error("DPU in tensor: " + dpu_in_name 
        + " not found in model runtime naming map.");
    }
    else 
    {
      in_tensor_order_.push_back(index);
    }
  }

  std::vector<std::string> rt_out_names;
  std::transform(out_tensor_names_.begin(), out_tensor_names_.end(),
   std::back_inserter(rt_out_names),
   [&rt_out_map](const std::string &elem)
   -> const std::string & { return rt_out_map[elem]; });

  for (int i = 0; i < out_tensor_names_.size(); ++i)
  {
    std::string dpu_out_name = dpu_runner_out_tensor_names[i];
    
    std::vector<std::string>::iterator iter = std::find_if(
      rt_out_names.begin(),
      rt_out_names.end(),
      [dpu_out_name](const std::string &elem) { return elem == dpu_out_name; });
    size_t index = std::distance(rt_out_names.begin(), iter);
    if (index == rt_out_names.size())
    {
      throw std::runtime_error("DPU out tensor: " + dpu_out_name 
        + " not found in model runtime naming map.");
    }
    else 
    {
      out_tensor_order_.push_back(index);
    }
  }

  pxDebug("Inside Initialize print in/out maps");
}

void DpuFunc::operator()(
  std::vector<XBufferHolder> &in_tensors,
  std::vector<XBufferHolder> &out_tensors)
{
  pxDebug("Inside VaiComputeFunc::()");

  std::vector<vitis::ai::CpuFlatTensorBuffer> inputs_cpu, outputs_cpu;
  std::vector<vitis::ai::TensorBuffer*> in_buffer, out_buffer;
  std::vector<std::shared_ptr<vitis::ai::Tensor>> batch_tensors;

  for (ssize_t i = 0; i < in_tensor_names_.size(); ++i)
  {
    batch_tensors.push_back(
      std::shared_ptr<vitis::ai::Tensor>(
        new vitis::ai::Tensor(dpu_runner_in_tensors_[i]->get_name(),
                              dpu_runner_in_tensors_[i]->get_dims(),
                              dpu_runner_in_tensors_[i]->get_data_type())
      )
    );

    inputs_cpu.push_back(
      vitis::ai::CpuFlatTensorBuffer(
        in_tensors[in_tensor_order_[i]]->data,
        batch_tensors.back().get())
    );
  }

  for (ssize_t i = 0; i < out_tensor_names_.size(); ++i)
  {
    batch_tensors.push_back(
      std::shared_ptr<vitis::ai::Tensor>(
        new vitis::ai::Tensor(dpu_runner_out_tensors_[i]->get_name(),
                              dpu_runner_out_tensors_[i]->get_dims(),
                              dpu_runner_out_tensors_[i]->get_data_type())
      )
    );
    outputs_cpu.push_back(
      vitis::ai::CpuFlatTensorBuffer(
        out_tensors[out_tensor_order_[i]]->data,
        batch_tensors.back().get())
    );
  }

  for(size_t i = 0; i < dpu_runner_in_tensors_.size(); ++i) {
    in_buffer.push_back(&inputs_cpu[i]);
  }

  for(size_t i = 0; i < dpu_runner_out_tensors_.size(); ++i) {
    out_buffer.push_back(&outputs_cpu[i]);
  }

  /*run*/
  auto start = std::chrono::high_resolution_clock::now();
  auto job_id = dpu_runner_->execute_async(in_buffer, out_buffer);
  auto stop1 = std::chrono::high_resolution_clock::now();
  dpu_runner_->wait(job_id.first,-1);
  auto stop2 = std::chrono::high_resolution_clock::now();


  std::cout << "exec_async Time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop1-start).count() << std::endl;
  std::cout << "wait Time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop2-stop1).count() << std::endl;
  std::cout << "total Time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop2-start).count() << std::endl;

}

} // vai_rt
} // namespace runtime
} // namespace pyxir