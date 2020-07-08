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

#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <sys/types.h>

namespace pyxir {

struct XBuffer {

  void* data;
  ssize_t itemsize;
  std::string format;
  ssize_t ndim;
  std::vector<ssize_t> shape;
  std::vector<ssize_t> strides;

  ssize_t size;
  bool own_data;

  XBuffer(const XBuffer &xb)
          : itemsize(xb.itemsize), format(xb.format), ndim(xb.ndim),
            shape(xb.shape), strides(xb.strides), size(xb.size),
            own_data(true)
  {
    // TODO: transfer data instead of copying
    // data = xb.data;
    // xb.own_data = false;
    data = ::operator new(size * itemsize);
    memcpy(data, xb.data, size * itemsize);
  }

  XBuffer(void *data_, ssize_t itemsize_, std::string format_, ssize_t ndim_,
          std::vector<ssize_t> shape_, std::vector<ssize_t> strides_,
          bool copy = true, bool own_data_ = true)
          : data(data_), itemsize(itemsize_), format(format_), ndim(ndim_),
            shape(shape_), strides(strides_), size(1), own_data(own_data_)
  {
    // std::cout << "Initialize XBuffer " << this << std::endl;
    if (ndim != (ssize_t) shape.size() || ndim != (ssize_t) strides.size())
      throw std::invalid_argument("XBuffer: ndim doesn't match shape and/or"
                                  " strides length");
    for (ssize_t i = 0; i < (ssize_t) ndim; ++i)
      size *= shape[i];

    if (copy) {
      data = ::operator new(size * itemsize);
      memcpy(data, data_, size * itemsize);
    } else {
      data = data_;
    }
  }

  XBuffer(void *data_, ssize_t itemsize_, std::string format_, ssize_t ndim_,
          std::vector<ssize_t> shape_, bool copy = true, bool own_data_ = true)
          : data(data_), itemsize(itemsize_), format(format_), ndim(ndim_),
            shape(shape_), size(1), own_data(own_data_)
  {
    if (ndim != (ssize_t) shape.size()) // || ndim != (ssize_t) strides.size()
      throw std::invalid_argument("XBuffer: ndim doesn't match shape and/or"
                                  " strides length");
    for (ssize_t i = 0; i < (ssize_t) ndim; ++i)
      size *= shape[i];

    ssize_t nxt_stride = (size > 0) ? size * itemsize : itemsize;
    strides.push_back(nxt_stride);
    for (ssize_t i = 1; i < (ssize_t) ndim; ++i) {
      nxt_stride /= shape[i];
      strides.push_back(nxt_stride);
    }

    if (copy) {
      data = ::operator new(size * itemsize);
      memcpy(data, data_, size * itemsize);
    } else {
      data = data_;
    }
  }

  ~XBuffer() {
    if (own_data) {
      ::operator delete(data);
    }
      
  }


};

typedef std::shared_ptr<XBuffer> XBufferHolder;

} // pyxir