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

#include <string>
#include <memory>


namespace pyxir {

struct StrContainer {
  
  std::string s_;

  StrContainer(const std::string &str = std::string()) : s_(str) {}
  StrContainer(const StrContainer &str_c) : s_(str_c.s_) {}

  std::string &get_string() { return s_; }

  void set_string(const std::string &s) { s_ = s; }

  StrContainer &operator=(const StrContainer &str_c)
  { 
    s_ = str_c.s_;
    return *this;
  }

};

typedef std::shared_ptr<StrContainer> StrContainerHolder;

struct BytesContainer {
  
  std::string s_;

  BytesContainer(const std::string &str = std::string()) : s_(str) {}
  BytesContainer(const BytesContainer &str_c) : s_(str_c.s_) {}

  std::string &get_string() { return s_; }

  void set_string(const std::string &s) { s_ = s; }

  BytesContainer &operator=(const BytesContainer &str_c)
  { 
    s_ = str_c.s_;
    return *this;
  }

};

typedef std::shared_ptr<BytesContainer> BytesContainerHolder;

} // pyxir
