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
#include "../graph/xgraph.hpp"

namespace py = pybind11;

namespace pyxir {
namespace graph {

void declare_xgraph(py::module &m) {

    py::class_<XGraph, std::shared_ptr<XGraph>>(m, "XGraph") // std::unique_ptr<XGraph, py::nodelete>
        .def(py::init<const std::string &>(),
             py::arg("name"))
        .def_readwrite("meta_attrs", &XGraph::meta_attrs)
        .def("copy", &XGraph::copy)
        .def("get_name", &XGraph::get_name)
        .def("set_name", &XGraph::set_name)
        .def("get_input_names", &XGraph::get_input_names)
        .def("get_output_names", &XGraph::get_output_names)
        .def("get", &XGraph::get,
             py::return_value_policy::reference_internal)
        .def("get_layer_names", &XGraph::get_layer_names)
        .def("__len__", &XGraph::len)
        .def("__contains__", &XGraph::contains)
        .def("add", &XGraph::add)
        .def("remove", &XGraph::remove)
        .def("update", &XGraph::update);
}

} // graph
} // pyxir