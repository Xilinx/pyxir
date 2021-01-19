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

#ifndef PX_API
#define PX_API __attribute__((visibility("default")))
#endif

#ifndef PX_UNUSED
#define PX_UNUSED __attribute__((visibility("default")))
#endif

#define STR_CONCAT_(__x, __y) __x##__y
#define STR_CONCAT(__x, __y) STR_CONCAT_(__x, __y)

inline void pxDebugMsg(const char *msg, const char *funcname,
                       const char *fname, int lineno)
{
  std::cout << "PYXIR(" << funcname << "): " << msg << " (" 
    << fname << ":" << lineno << ")" << std::endl;
}

inline void pxWarningMsg(const char * msg, const char *fname, int lineno)
{
  std::cout << "PYXIR[WARNING]: " << msg << " (" 
    << fname << ":" << lineno << ")" << std::endl;
}

inline void pxWarningMsg(const std::string &msg, const char *fname, int lineno)
{
  pxWarningMsg(msg.c_str(), fname, lineno);
}

inline void pxInfoMsg(const char * msg)
{
  std::cout << "PYXIR[INFO]: " << msg << std::endl;
}

inline void pxInfoMsg(const std::string &msg)
{
  pxInfoMsg(msg.c_str());
}

#define pxWarning(x) pxWarningMsg(x, __FILE__, __LINE__);
#define pxInfo(x) pxInfoMsg(x);

#ifdef DEBUG
#define pxDebug(x) pxDebugMsg(x, __FUNCTION__, __FILE__, __LINE__);
#else
#define pxDebug(x)
#endif

/**
 * @brief Return whether verbose flag was set and is true
 */
inline bool is_verbose() {
  const char* px_verbose_flag = std::getenv("PX_VERBOSE");
  if (px_verbose_flag) {
    std::string verbose = std::string(px_verbose_flag);
    return verbose == "True" || verbose == "true" || verbose == "1";
  }
  return false;
}