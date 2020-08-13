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
#include "../common/xbuffer.hpp"

namespace py = pybind11;

namespace pyxir {

void declare_xbuffer(py::module &m) {

	py::class_<XBuffer, std::shared_ptr<XBuffer>>(
		  m, "XBuffer", py::buffer_protocol())
		.def(py::init([](py::buffer b) {

			/* Request a buffer descriptor from Python */
			py::buffer_info info = b.request();

			return new XBuffer(info.ptr, info.itemsize, info.format,
                               info.ndim, info.shape, info.strides);
		}))
		.def_buffer([](XBuffer &xb) -> py::buffer_info {
			return py::buffer_info(
					xb.data,        /* Pointer to data */
					xb.itemsize,    /* Size of one item in bytes */
					xb.format,      /* The format descriptor */
					xb.ndim,        /* Number of dimensions */
					xb.shape,       /* Buffer dimensions */
					xb.strides      /* Strides (in bytes) for each index */
			);
		});
}

} // pyxir