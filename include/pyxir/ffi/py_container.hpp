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
#include "str_container.hpp"

namespace py = pybind11;

namespace pyxir {

void declare_containers(py::module &m) {

  py::class_<StrContainer, std::shared_ptr<StrContainer>>(m, "StrContainer")
    .def(py::init<const std::string &>(),
         py::arg("str") = std::string())
    .def_readwrite("str", &StrContainer::s_);

  py::class_<BytesContainer, std::shared_ptr<BytesContainer>>(m, "BytesContainer")
    .def(py::init<const py::bytes &>(),
         py::arg("str") = py::bytes())
    .def("set_bytes", [](BytesContainer & sc, const py::bytes &b) {
      sc.s_ = b;
    })
    .def("get_bytes", [](BytesContainer &sc) {
      return py::bytes(sc.s_);
    });

};

} // pyxir