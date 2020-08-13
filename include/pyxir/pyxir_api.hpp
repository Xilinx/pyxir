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

inline void pxDebugMsg(const char * msg, const char *funcname,
                       const char *fname, int lineno)
{
  std::cout << "PYXIR(" << funcname << "): " << msg << " (" 
    << fname << ":" << lineno << ")" << std::endl;
}

inline void pxWarningMsg(const char * msg, const char *funcname,
                         const char *fname, int lineno)
{
  std::cout << "PYXIR[WARNING]: " << msg << " (" 
    << fname << ":" << lineno << ")" << std::endl;
}

inline void pxWarningMsg(const std::string &msg, const char *funcname,
                         const char *fname, int lineno)
{
  pxWarningMsg(msg.c_str(), funcname, fname, lineno);
}

#define pxWarning(x) pxWarningMsg(x, __FUNCTION__, __FILE__, __LINE__);

#ifdef DEBUG
#define pxDebug(x) pxDebugMsg(x, __FUNCTION__, __FILE__, __LINE__);
#else
#define pxDebug(x)
#endif