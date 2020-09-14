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
#include <sstream>

#include "../pyxir_api.hpp"
#include "../graph/xgraph.hpp"

namespace pyxir {
namespace onnx {

/**
 * @brief Import ONNX Python APIs
 */
PX_API void import_py_onnx();

/**
 * @brief Import ONNX model from the given input file
 * @param file_path The location of the input file
 * @returns An XGraph representation of the provided ONNX model
 */
PX_API std::shared_ptr<graph::XGraph> import_onnx_model(
  const std::string &file_path);

/**
 * @brief Import ONNX model from input string stream
 * @param sstream The input string stream
 * @returns An XGraph representation of the provided ONNX model
 */
PX_API std::shared_ptr<graph::XGraph> import_onnx_model(
  std::istringstream &sstream);

} // onnx
} // pyxir
