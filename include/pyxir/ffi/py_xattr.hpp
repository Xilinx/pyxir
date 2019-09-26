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
#include "../graph/xattr.hpp"

namespace py = pybind11;

namespace pyxir {
namespace graph {

void declare_xattr(py::module &m) {

    py::class_<XAttr>(m, "XAttr")
        .def(py::init<const std::string &>(),
             py::arg("name"))
        .def(py::init<const std::string &, bool>(),
             py::arg("name"),
             py::arg("b"))
        .def(py::init<const std::string &, int>(),
             py::arg("name"),
             py::arg("i"))
        .def(py::init<const std::string &, const std::vector<int64_t>>(),
             py::arg("name"),
             py::arg("ints"))
        .def(py::init<const std::string &,
                      const std::vector<std::vector<int64_t>>>(),
             py::arg("name"),
             py::arg("ints2d"))
        .def(py::init<const std::string &, double>(),
             py::arg("name"),
             py::arg("f"))
        .def(py::init<const std::string &, const std::vector<double>>(),
             py::arg("name"),
             py::arg("floats"))
        .def(py::init<const std::string &, const std::string &>(),
             py::arg("name"),
             py::arg("s"))
        .def(py::init<const std::string &, const std::vector<std::string>>(),
             py::arg("name"),
             py::arg("strings"))\
        .def(py::init<const std::string &, const std::unordered_map<std::string, std::string> &>(),
             py::arg("name"),
             py::arg("map_str_str"))
        .def(py::init<const std::string &, const std::unordered_map<std::string, std::vector<std::string>>>(),
             py::arg("name"),
             py::arg("map_str_vstr"))
        .def_readwrite("name", &XAttr::name)
        .def_readwrite("type", &XAttr::type)
        .def_readwrite("b", &XAttr::b)
        .def_readwrite("i", &XAttr::i)
        .def_property("ints", &XAttr::get_ints, &XAttr::set_ints)
        .def_property("ints2d", &XAttr::get_ints2d, &XAttr::set_ints2d)
        .def_readwrite("f", &XAttr::f)
        .def_property("floats", &XAttr::get_floats, &XAttr::set_floats)
        .def_property("s", &XAttr::get_string, &XAttr::set_string)
        .def_property("strings", &XAttr::get_strings, &XAttr::set_strings)
        .def_property("map_str_str", &XAttr::get_map_str_str, 
                      &XAttr::set_map_str_str)
        .def_property("map_str_vstr", &XAttr::get_map_str_vstr, 
                      &XAttr::set_map_str_vstr);

}

} // graph
} // pyxir
