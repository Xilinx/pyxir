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

#include "../common/px_stream.hpp"
#include "../graph/xgraph.hpp"
#include "../pyxir_api.hpp"

namespace pyxir {

/**
 * @brief Write the XGraph to a string stream
 * @param xg The XGraph to be written to the string stream
 * @param sstream The output string stream for the graph to be written to
 */
PX_API void write(XGraphHolder &xg, PxOStringStream &sstream);

/**
 * @brief Write the XGraph to a string stream
 * @param xg The XGraph to be written to the string stream
 * @param sstream The output string stream for the graph to be written to
 */
PX_API void write(XGraphHolder &xg, std::ostringstream &sstream);

/**
 * @brief Read an XGraph from a string stream
 * @param xg The XGraph to be initialized from the stream
 * @param sstream The input string stream for the graph to be read from
 */
PX_API void read(XGraphHolder &xg, PxIStringStream &sstream);

/**
 * @brief Read an XGraph from a string stream
 * @param xg The XGraph to be initialized from the stream
 * @param sstream The input string stream for the graph to be read from
 */
PX_API void read(XGraphHolder &xg, std::istringstream &sstream);

} // namespace pyxir
