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

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

// #include "../opaque_func.hpp"
#include "../opaque_func_registry.hpp"

namespace py = pybind11;

namespace pyxir {

void declare_opaque_func(py::module &m) {

  // py::bind_vector<std::vector<OpaqueValue>>(
  //     m, "OpaqueValueVector", py::module_local(false));

  py::class_<OpaqueValue, std::shared_ptr<OpaqueValue>>(m, "OpaqueValue")
    .def(py::init<const std::vector<int64_t> &>(),
         py::arg("ints"))
    .def(py::init<const std::string &>(),
         py::arg("s"))
    .def(py::init<const std::vector<std::string> &>(),
         py::arg("strings"))
    .def(py::init<StrContainerHolder &>(),
         py::arg("str_c"))
    .def(py::init<BytesContainerHolder &>(),
         py::arg("bytes_c"))
    .def(py::init<std::shared_ptr<graph::XGraph> &>(),
         py::arg("xg"))
    .def(py::init<std::shared_ptr<XBuffer> &>(),
         py::arg("xb"))
    .def(py::init<const std::vector<std::shared_ptr<XBuffer>> &>(),
         py::arg("xbuffers"))
    .def(py::init<std::shared_ptr<OpaqueFunc> &>(),
         py::arg("of"))
    .def("get_type_code_str", &OpaqueValue::get_type_code_str)
    .def("get_type_code_int", &OpaqueValue::get_type_code_int)
    .def_property("ints", &OpaqueValue::get_ints, &OpaqueValue::set_ints)
    .def_property("s", &OpaqueValue::get_string, &OpaqueValue::set_string)
    .def_property("bytes", [](OpaqueValue &ov) {
                               return py::bytes(ov.get_string()); },
                           &OpaqueValue::set_string)
    .def_property("strings", &OpaqueValue::get_strings, &OpaqueValue::set_strings)
    .def_property("str_c", &OpaqueValue::get_str_container, &OpaqueValue::set_str_container)
    .def_property("bytes_c", &OpaqueValue::get_bytes_container, &OpaqueValue::set_bytes_container)
    .def_property("xg", &OpaqueValue::get_xgraph, &OpaqueValue::set_xgraph)
    .def_property("xb", &OpaqueValue::get_xbuffer, &OpaqueValue::set_xbuffer)
    .def_property("xbuffers", &OpaqueValue::get_xbuffers, &OpaqueValue::set_xbuffers)
    .def_property("of", &OpaqueValue::get_opaque_func, &OpaqueValue::set_opaque_func);

  py::class_<OpaqueArgs, std::shared_ptr<OpaqueArgs>>(m, "OpaqueArgs")
    .def(py::init<std::vector<std::shared_ptr<OpaqueValue>> &>(),
         py::arg("args"))
    .def("__len__", &OpaqueArgs::size)
    .def("__getitem__", &OpaqueArgs::operator[]);

  py::class_<OpaqueFunc, std::shared_ptr<OpaqueFunc>>(m, "OpaqueFunc")
    .def(py::init<>())
    .def(py::init<const std::function<void (OpaqueArgs &)> &,
                  const std::vector<int64_t> &>(),
         py::arg("func"),
         py::arg("arg_type_codes"))
    .def("get_func", &OpaqueFunc::get_func)
    .def("set_func", (void (OpaqueFunc::*)(
                        const std::function<void (OpaqueArgs &)> &,
                        const std::vector<int64_t> &))
                      &OpaqueFunc::set_func)
    .def("get_arg_type_codes", &OpaqueFunc::get_arg_type_codes)
    .def("__call__", &OpaqueFunc::call);

  /// @brief Create bindings for the OpaqueFuncRegistry
  py::class_<OpaqueFuncRegistry, std::shared_ptr<OpaqueFuncRegistry>>(
    m, "OpaqueFuncRegistry")
    .def(py::init<>())
    .def("set_func", (OpaqueFuncRegistry& 
                       (OpaqueFuncRegistry::*)(OpaqueFunc &)) 
                      &OpaqueFuncRegistry::set_func)
    .def("get_func", &OpaqueFuncRegistry::get_func)
    //.def("set_args_type_codes", &OpaqueFuncRegistry::set_args_type_codes)
    //.def("get_args_type_codes", &OpaqueFuncRegistry::get_args_type_codes)
    .def_static("Register", &OpaqueFuncRegistry::Register)
    .def_static("Exists", &OpaqueFuncRegistry::Exists)
    .def_static("Get", &OpaqueFuncRegistry::Get)
    //.def_static("GetArgsTypeCodes", &OpaqueFuncRegistry::GetArgsTypeCodes)
    .def_static("GetRegisteredFuncs", &OpaqueFuncRegistry::GetRegisteredFuncs)
    .def_static("Size", &OpaqueFuncRegistry::Size)
    .def_static("Clear", &OpaqueFuncRegistry::Clear);
}

} // pyxir
