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

#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/embed.h>

#include "pyxir/pyxir.hpp"
#include "pyxir/ffi/py_container.hpp"
#include "pyxir/ffi/py_xlayer.hpp"
#include "pyxir/ffi/py_xbuffer.hpp"
#include "pyxir/ffi/py_xattr.hpp"
#include "pyxir/ffi/py_xgraph.hpp"
#include "pyxir/ffi/py_opaque_func.hpp"


// #ifdef USE_TEST_FEATURE
// int TEST_FEATURE=1;
// #else
// int TEST_FEATURE=0;
// #endif

// using namespace pyxir;
using namespace pyxir::graph;
using namespace std::chrono; 

int add(int a, int b) {
  return a + b;
};


PYBIND11_MAKE_OPAQUE(std::vector<int64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(std::vector<pyxir::XBuffer>);
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<pyxir::XBuffer>>);
// PYBIND11_MAKE_OPAQUE(std::vector<XLayer>);

PYBIND11_MAKE_OPAQUE(std::vector<std::vector<int64_t>>);

PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, XAttr>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, std::vector<std::string>>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, std::string>);

PYBIND11_MODULE(libpyxir, m) {
    m.doc() = R"pbdoc(
        PyXIR C++ library
        -----------------
    )pbdoc";

    py::bind_vector<std::vector<int64_t>>(
      m, "IntVector", py::module_local(false));
    py::bind_vector<std::vector<double>>(
      m, "FloatVector", py::module_local(false));
    py::bind_vector<std::vector<std::string>>(
      m, "StrVector", py::module_local(false));
    py::bind_vector<std::vector<pyxir::XBuffer>>(
      m, "XBufferVector", py::module_local(false));
    py::bind_vector<std::vector<std::shared_ptr<pyxir::XBuffer>>>(
      m, "XBufferHolderVector", py::module_local(false));
    // py::bind_vector<std::vector<XLayer>>(
    //   m, "XLayerVector", py::module_local(false));

    py::bind_vector<std::vector<std::vector<int64_t>>>(
      m, "IntVector2D", py::module_local(false));

    py::bind_map<std::unordered_map<std::string, XAttr>>(
      m, "XAttrMap", py::module_local(false));

    py::bind_map<std::unordered_map<std::string, std::string>>(
      m, "MapStrStr", py::module_local(false));
    py::bind_map<std::unordered_map<std::string, std::vector<std::string>>>(
      m, "MapStrVectorStr", py::module_local(false));

    pyxir::declare_opaque_func(m);

    pyxir::declare_xbuffer(m);
    pyxir::declare_containers(m);
    pyxir::graph::declare_xattr(m);
    pyxir::graph::declare_xlayer(m);
    pyxir::graph::declare_xgraph(m);

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

    auto cleanup_callback = []() {
        // perform cleanup here -- this function is called with the GIL held
        // std::cout << "Pybind11 cleanup" << std::endl;

        // Make sure registry gets deleted in the right order?!
        // TODO: Better solution??
        pyxir::OpaqueFuncRegistry::Clear();
    };

    m.add_object("_cleanup", py::capsule(cleanup_callback));


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}