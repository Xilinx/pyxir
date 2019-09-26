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
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include "../graph/xlayer.hpp"

namespace py = pybind11;

namespace pyxir {
namespace graph {

void declare_xlayer(py::module &m) {

    py::class_<XLayer, std::shared_ptr<XLayer>>(m, "XLayer")
        //.def(py::init<>())
        .def(py::init<const std::string &, 
                      const std::vector<std::string> &,
                      const std::vector<std::vector<int64_t>> &,
                      const std::string &,
                      const std::vector<int64_t> &,
                      const std::vector<std::string> &,
                      const std::vector<std::string> &,
                      const std::vector<std::string> &,
                      const std::vector<XBuffer> &,
                      const std::vector<std::string> &,
                      const std::string &,
                      const std::string &,
                      const bool,
                      const std::unordered_map<std::string, XAttr> &>(),
             py::arg("name") = std::string(),
             py::arg("xtype") = std::vector<std::string>(),
             py::arg("shapes") = std::vector<std::vector<int64_t>>(),
             py::arg("shapes_t") = std::string("TensorShape"),
             py::arg("sizes") = std::vector<int64_t>(),
             py::arg("bottoms") = std::vector<std::string>(),
             py::arg("tops") = std::vector<std::string>(),
             py::arg("layer") = std::vector<std::string>(),
             py::arg("data") = std::vector<XBuffer>(),
             py::arg("targets") = std::vector<std::string>(),
             py::arg("target") = std::string(),
             py::arg("subgraph") = std::string(),
             py::arg("internal") = false,
             py::arg("attrs") = std::unordered_map<std::string, XAttr>())
        .def_readwrite("name", &XLayer::name)
        .def_readwrite("xtype", &XLayer::xtype)
        .def_readwrite("shapes", &XLayer::shapes)
        .def_readwrite("shapes_t", &XLayer::shapes_t)
        .def_readwrite("sizes", &XLayer::sizes)
        .def_readwrite("bottoms", &XLayer::bottoms)
        .def_readwrite("tops", &XLayer::tops)
        .def_readwrite("layer", &XLayer::layer)
        .def_property("data", &XLayer::get_data, &XLayer::set_data)
        .def_readwrite("targets", &XLayer::targets)
        .def_readwrite("target", &XLayer::target)
        .def_readwrite("subgraph", &XLayer::subgraph)
        // .def("get_subgraph_data", &XLayer::get_subgraph_data,
        //      py::return_value_policy::reference_internal)
        // .def("set_subgraph_data", 
        //      (void (XLayer::*)(const std::vector<XLayer> &)) 
        //      &XLayer::set_subgraph_data)
        .def_property("subgraph_data", &XLayer::get_subgraph_data, 
          (void (XLayer::*)(const std::vector<XLayer> &)) 
          &XLayer::set_subgraph_data)
        .def_readwrite("internal", &XLayer::internal)
        .def_readwrite("attrs", &XLayer::attrs);
};

} // graph
} // pyxir