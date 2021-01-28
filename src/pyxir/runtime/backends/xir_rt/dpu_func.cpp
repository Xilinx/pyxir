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
#include "common.h"

#include "dpu_func.hpp"
GraphInfo shapes;


namespace pyxir {
namespace runtime {
namespace vai_rt {

DpuFunc::DpuFunc(XLayerHolder &xl, const std::string &build_dir) : KernelFunc(xl)
{
  std::vector<std::string> dpu_in_tensor_names = xl->bottoms;
  std::vector<std::string> dpu_internal_in_tensor_names
    = xl->get_attr("input_names").get_strings();
  
  in_tensor_names_ = dpu_internal_in_tensor_names;

  std::vector<std::string> dpu_out_tensor_names = xl->get_attr("output_names").get_strings(); // xl->tops;
  out_tensor_names_ = dpu_out_tensor_names;
  
  std::unordered_map<std::string, std::string> rt_in_map = 
    xl->get_attr("rt_in_map").get_map_str_str();
  std::unordered_map<std::string, std::string> rt_out_map = 
    xl->get_attr("rt_out_map").get_map_str_str();
  
  pxDebug("Before DpuRunner init");
  // Setup DPU runner
  std::string model_path;
  if (!build_dir.empty()) {
    model_path = build_dir;
  } else {
    model_path = xl->get_attr("work_dir").get_string();
  }
  graph = xir::Graph::deserialize(model_path+"/xp0.xmodel");
  
  //graph = xir::Graph::deserialize(build_dir+"/xp0.xmodel");
  subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "resnet50 should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
  /*create runner*/
  runner = vart::Runner::create_runner(subgraph[0], "run");
/*
  auto dpu_runners = vitis::ai::DpuRunner::create_dpu_runner(model_path);
  dpu_runner_ = std::move(dpu_runners[0]);

  if(dpu_runner_->get_tensor_format() != vitis::ai::DpuRunner::TensorFormat::NCHW) {
    pxDebug("Invalid tensor format NHWC");
  }
  pxDebug("After DpuRunner init");
 */
  dpu_runner_in_tensors_ = runner->get_input_tensors();
  dpu_runner_out_tensors_ = runner->get_output_tensors();
  assert(dpu_runner_in_tensors_.size() == dpu_in_tensor_names.size());
  assert(dpu_runner_out_tensors_.size() == dpu_out_tensor_names.size());

  std::vector<std::string> dpu_runner_in_tensor_names;
  std::transform(dpu_runner_in_tensors_.begin(), dpu_runner_in_tensors_.end(),
    std::back_inserter(dpu_runner_in_tensor_names),
    [](const xir::Tensor* t) -> const std::string { return t->get_name(); });
 // dpu_runner_in_tensor_names[0]="xinput0";
  std::vector<std::string> dpu_runner_out_tensor_names;

  std::transform(dpu_runner_out_tensors_.begin(), dpu_runner_out_tensors_.end(),
   std::back_inserter(dpu_runner_out_tensor_names),
   [](const xir::Tensor* t) -> const std::string { return t->get_name(); });


  string name= "/aquant";
  for(int i=0;i< dpu_runner_in_tensor_names.size();i++)
  {
	  size_t pos = dpu_runner_in_tensor_names[i].find(name);
	  dpu_runner_in_tensor_names[i].replace(pos,name.length(),"");
  }
  for(int i=0;i< dpu_runner_out_tensor_names.size();i++)
  {
	  size_t pos = dpu_runner_out_tensor_names[i].find(name);
	  dpu_runner_out_tensor_names[i].replace(pos,name.length(),"");
  }


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

DpuFunc::~DpuFunc() {
  if (is_verbose()) {
    std::cout << "---------------------" << std::endl;
    std::cout << "PX DPU FUNC TIMINGS: " << std::endl;
    std::cout << "Total DPU time: " << std::to_string(total_dpu_time_) << std::endl;
    std::cout << "Total async time: " << std::to_string(total_async_time_) << std::endl;
    std::cout << "Total wait time: " << std::to_string(total_wait_time_) << std::endl;
    std::cout << "---------------------" << std::endl;
  }
}

void DpuFunc::operator()(
  std::vector<XBufferHolder> &in_tensors,
  std::vector<XBufferHolder> &out_tensors)
{
 /*get in/out tensor*/
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;
  int in_idx = 0;
  for(const auto& iTensor: inputTensors) {
	  const auto& in_dims = iTensor->get_shape();
	  batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(iTensor->get_name(), in_dims, xir::DataType{xir::DataType::FLOAT, sizeof(float) * 8u})));
    inputsPtr.push_back(new CpuFlatTensorBuffer(in_tensors[in_idx]->data, batchTensors.back().get()));
    in_idx++;
  }

  if (out_tensors.empty()) {
    for (const auto &shape : xl_->shapes) {
      std::vector<ssize_t> buffer_shape = shape;
      buffer_shape[0] = inputTensors[0]->get_shape()[0];
      //buffer_shape[0] = in_tensors[0]->shape[0];
      out_tensors.push_back(create_buffer(buffer_shape));
    }
  }
  
 int out_idx=0;
 for(const auto& oTensor: outputTensors) {
    const auto& out_dims = oTensor->get_shape();
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(oTensor->get_name(), out_dims, xir::DataType{xir::DataType::FLOAT, sizeof(float) * 8u})));
    outputsPtr.push_back(new CpuFlatTensorBuffer(out_tensors[out_tensor_order_[out_idx]]->data, batchTensors.back().get()));
    //outputsPtr.push_back(new CpuFlatTensorBuffer(out_tensors[out_idx]->data, batchTensors.back().get()));
     out_idx++;

}
  LOG(INFO) << "Executing       ";
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);

  for(int i = 0; i<inputsPtr.size(); ++i) {
    delete inputsPtr[i];
  }

  for(int i = 0; i<outputsPtr.size(); ++i) {
    delete outputsPtr[i];  
}        

}

} // vai_rt
} // namespace runtime
} // namespace pyxir

