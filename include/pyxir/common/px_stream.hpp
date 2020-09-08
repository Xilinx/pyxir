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
#include <sstream>

namespace pyxir {

// void to_bitset(std::string &str, istringstream &sstream)
// {
//   for (char &c : str) {
//     bit_set_v.pushback(bitset<8>(c).to_string());
//   }
// }


// std::string from_binary_string(std::string &bin_str)
// {
//   std::string str = "";
//   for (char &c : bin_str) {
//       str += bitset<8>(c).to_string();
//   }
//   return str;
// }


class PxIStringStream {

  public:
    PxIStringStream(std::istringstream &sstream) : sstream_(sstream) {}

    std::istringstream &get_istringstream() const { return sstream_; }

    void read(std::string &str)
    {
      int size;
      sstream_ >> size;
      auto p = sstream_.tellg();
      sstream_.seekg(p + (std::streamoff) 1);
      str.resize(size, '\0');
      sstream_.read(&str[0], size);

      if (sstream_.bad())
        throw std::runtime_error("I/O error while reading");
      else if (sstream_.fail())
        throw std::runtime_error("Reading string from istringstream failed");
    }

    template <typename T>
    void read(T &t)
    {
      int size;
      sstream_ >> size >> t;
    }

  private:
    std::istringstream &sstream_;
};

class PxOStringStream {

  public:
    PxOStringStream(std::ostringstream &sstream) : sstream_(sstream) {}

    std::ostringstream &get_ostringstream() const { return sstream_; }

    void write(const std::string &str)
    {
      sstream_ << " " << std::to_string(str.size()) << " " << str;
    }

    void write(const char *str)
    {
      std::string s = str;
      write(s);
    }

    template <typename T>
    void write(const T &b)
    {
      write(std::to_string(b));
    }

  private:
    std::ostringstream &sstream_;
};

} // pyxir 