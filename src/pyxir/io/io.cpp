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

#include <memory>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include "pyxir/ffi/str_container.hpp"
#include "pyxir/frontend/onnx.hpp"
#include "pyxir/opaque_func_registry.hpp"
#include "pyxir/common/px_stream.hpp"

namespace py = pybind11;

namespace pyxir {

void write(XGraphHolder &xg, PxOStringStream &sstream)
{
  if (!pyxir::OpaqueFuncRegistry::Exists("pyxir.io.to_string"))
    throw std::runtime_error("Cannot write XGraph to string stream because"
                             " `pyxir.io.to_string` opaque function is"
                             " not registered");
  
  OpaqueFunc to_string = 
    pyxir::OpaqueFuncRegistry::Get("pyxir.io.to_string");
  
  BytesContainerHolder graph_str = BytesContainerHolder(new BytesContainer());
  BytesContainerHolder data_str = BytesContainerHolder(new BytesContainer());
  to_string(xg, graph_str, data_str);

  sstream.write(graph_str->s_);
  sstream.write(data_str->s_);
}

PX_API void write(XGraphHolder &xg, std::ostringstream &sstream)
{
  PxOStringStream pxoss(sstream);
  write(xg, pxoss);
}

void read(XGraphHolder &xg, PxIStringStream &sstream)
{
  if (!pyxir::OpaqueFuncRegistry::Exists("pyxir.io.from_string"))
    throw std::runtime_error("Cannot create XGraph from string stream because"
                             " `pyxir.io.from_string` opaque function is"
                             " not registered");
  
  OpaqueFunc from_string = 
    pyxir::OpaqueFuncRegistry::Get("pyxir.io.from_string");

  // XGraphHolder xg = std::make_shared<pyxir::graph::XGraph>("empty_xgraph");
  
  std::string graph_str;
  std::string data_str;
  sstream.read(graph_str);
  sstream.read(data_str);
  from_string(xg, graph_str, data_str);
}

PX_API void read(XGraphHolder &xg, std::istringstream &sstream)
{
  PxIStringStream pxiss(sstream);
  read(xg, pxiss);
}

REGISTER_OPAQUE_FUNC("pyxir.io.get_serialized_xgraph")
    ->set_func([](pyxir::OpaqueArgs &args) 
    {
      XGraphHolder xg = args[0]->get_xgraph();
      std::ostringstream sstream;
      write(xg, sstream);
      args[1]->get_bytes_container()->set_string(sstream.str());
    }, std::vector<pxTypeCode>{pxXGraphHandle, pxBytesContainerHandle});


REGISTER_OPAQUE_FUNC("pyxir.io.deserialize_xgraph")
    ->set_func([](pyxir::OpaqueArgs &args) 
    {
      XGraphHolder xg = args[0]->get_xgraph();
      std::istringstream sstream(args[1]->get_bytes_container()->get_string());
      read(xg, sstream);
    }, std::vector<pxTypeCode>{pxXGraphHandle, pxBytesContainerHandle});

} // pyxir