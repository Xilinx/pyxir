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
 #include <fstream>

#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>



#include "pyxir/common/util.hpp"
#include "dpu_func_int8.hpp"
using namespace std;
namespace pyxir {
namespace runtime {
namespace vai_rt {
void DpuFuncInt8::operator()(
  std::vector<XBufferHolder> &in_tensors,
  std::vector<XBufferHolder> &out_tensors)
{
  auto runner = runner_.get();
  auto inputs = dynamic_cast<vart::RunnerExt*>(runner)->get_inputs();
  auto outputs = dynamic_cast<vart::RunnerExt*>(runner)->get_outputs();

  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  auto & out_dims = outputTensors[0]->get_shape();
  auto & in_dims = inputTensors[0]->get_shape();

  auto scale_in = pow(2,(*inputs.begin())->get_tensor()->get_attr<std::int32_t>("fix_point"));
  std::vector<float> scale_out;

	int8_t* std_data = reinterpret_cast<int8_t*>(inputs[0]->data().first);
  std::vector<int8_t*> std_data_out;
  auto inSize = inputs[0]->get_tensor()->get_element_num();
  std::vector<int32_t> outSize;
  auto pData = static_cast<float*>(in_tensors[0]->data);
  int out_idx = 0;

  for (const auto &oTensor : outputTensors)
  {
    std_data_out.push_back(reinterpret_cast<int8_t*>(outputs[out_idx]->data().first));
    outSize.push_back(outputs[out_idx]->get_tensor()->get_element_num());
    scale_out.push_back(pow(2,(-1)*(oTensor->get_attr<std::int32_t>("fix_point"))));
    out_idx++;
  }

  for(auto i = 0; i < inSize; i++)
  {
    std_data[i] = static_cast<int8_t>(pData[i] * scale_in);
  }

  auto job_id = runner->execute_async(inputs, outputs);
  runner->wait(job_id.first, -1);

 out_idx = 0;

  std::vector<float *> out_pyxir;
  for (const auto &oTensor : outputTensors)
  {
    out_pyxir.push_back((float*)out_tensors_local_[out_tensor_order_[out_idx]]->data);
    for (int i=0;i < outSize[out_idx] ; i++)
      {
        int8_t fix = std_data_out[out_idx][i];
        out_pyxir[out_idx][i] = ((float) fix) * scale_out[out_idx];
      }
    out_idx++;
  }
  out_idx = 0;
  if (out_tensors.empty())
  {
    for (const auto &shape : xl_->shapes)
    {
      std::vector<ssize_t> buffer_shape = shape;
      buffer_shape[0] = in_tensors[0]->shape[0];

      pyxir::XBufferHolder xb_out = std::shared_ptr<pyxir::XBuffer>(new pyxir::XBuffer(
                                      (void *)out_tensors_local_[out_idx]->data, 4,
                                      "f", buffer_shape.size(), buffer_shape, true, true));

      out_tensors.push_back(xb_out);
      out_idx++;
    }
  }
}

} // vai_rt
} // namespace runtime
} // namespace pyxir

