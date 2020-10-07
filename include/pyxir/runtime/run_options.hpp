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

#include <cstdlib>
#include <memory>
#include <bitset>

#include "../common/serializable.hpp"

namespace pyxir {
namespace runtime {

struct RunOptions : public ISerializable {
  
  RunOptions() {
    const char *env_quant_size = std::getenv("PX_QUANT_SIZE");
    if (env_quant_size != NULL)
      nb_quant_inputs = std::atoi(env_quant_size);
  }

  /** @brief Whether to use on-the-fly quantization */
  bool on_the_fly_quantization = false;
  /** @brief The number of inputs to be used for quantization (= calibration dataset) */
  int nb_quant_inputs = 128;
  /** @brief The location of the final build directory */
  std::string build_dir = "/tmp/vitis_ai_build";
  /** @brief The location of the directory for temporary work files */
  std::string work_dir = "/tmp/vitis_ai_work";
  /** @brief Whether the build has been completed, allows use of pre-built build directory */
  bool is_prebuilt = false;
  /** @brief Export runtime module somewhere after build, the specific compute
        implementation is free to choose when this happens. */
  std::string export_runtime_module_path = "";
  /** @brief Load runtime module somewhere after build, the specific compute
        implementation is free to choose when this happens. */
  // std::string load_runtime_module_path = ""; 

  virtual void serialize_px(PxOStringStream &pstream)
  {
    pstream.write(on_the_fly_quantization);
    pstream.write(nb_quant_inputs);
    pstream.write(build_dir);
    pstream.write(work_dir);
    pstream.write(is_prebuilt);
    pstream.write(export_runtime_module_path);
  }

  virtual void deserialize_px(PxIStringStream &pstream)
  {
    pstream.read(on_the_fly_quantization);
    pstream.read(nb_quant_inputs);
    pstream.read(build_dir);
    pstream.read(work_dir);
    pstream.read(is_prebuilt);
    pstream.read(export_runtime_module_path);
  }
};

} // namespace runtime

typedef std::shared_ptr<runtime::RunOptions> RunOptionsHolder;

} // namespace pyxir
