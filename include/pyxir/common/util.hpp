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

#include <iostream>
#include <string>
#include <regex>
#include <ftw.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

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

inline bool is_dir(const std::string &dir_path)
{
  struct stat info;
  return stat(dir_path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

// From https://stackoverflow.com/questions/5467725/how-to-delete-a-directory-and-its-contents-in-posix-c
inline int unlink_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf)
{
  // Silence warnings
  (void) sb;
  (void) typeflag;
  (void) ftwbuf;

  int rv = remove(fpath);
  if (rv)
    perror(fpath);

  return rv;
}

inline void rmrf(const std::string &path)
{
  if (nftw(path.c_str(), unlink_cb, 64, FTW_DEPTH | FTW_PHYS) == -1) {
    // char *error = std::strerror(errno);
    // std::string str_error(error, std::find(error, '\0'));
    // throw std::runtime_error("Error :  " + str_error);
    std::cerr << "Error :  " << strerror(errno) << " : " << path.c_str()<< std::endl;
  }
}

} // pyxir
