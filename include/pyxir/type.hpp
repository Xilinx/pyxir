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

/**
 * @brief The type code used for identifying opaque object types
 */
typedef enum {
  pxInt,
  pxVInt,
  pxFloat,
  pxVFloat,
  pxStrHandle,
  pxVStrHandle,
  pxStrContainerHandle,
  pxBytesContainerHandle,
  pxXGraphHandle,
  pxXBufferHandle,
  pxVXBufferHandle,
  pxOpaqueFuncHandle,
  pxUndefined
} pxTypeCode;


inline const std::string px_type_code_to_string(pxTypeCode ptc) 
{
  std::cout << "px type code to string: " << ptc << std::endl;
  switch (ptc)
  {
    case pxInt: return "Int";
    case pxVInt: return "vInt";
    case pxFloat: return "Float";
    case pxVFloat: return "vFloat";
    case pxStrHandle: return "Str";
    case pxVStrHandle: return "vStr";
    case pxStrContainerHandle: return "StrC";
    case pxBytesContainerHandle: return "BytesC";
    case pxXGraphHandle: return "XGraph";
    case pxXBufferHandle: return "XBuffer";
    case pxVXBufferHandle: return "vXBuffer";
    case pxOpaqueFuncHandle: return "OpaqueFunc";
    case pxUndefined: return "Undefined";
    default: 
      throw std::invalid_argument("Unkwown type code to string function");
  }
}