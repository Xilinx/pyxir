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
#include <regex>

namespace pyxir {

inline bool is_str_number(const std::string& s)
{
	return !s.empty() && std::find_if(s.begin(), 
		s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

inline std::string stringify(const std::string &s) {
  std::regex vowel_re("[^A-Za-z0-9_.\\->/]");
  std::string s2 = std::regex_replace(s, vowel_re, "-");
  if (is_str_number(s2))
    return s2 + "_";
  return s2;
}

} // pyxir
