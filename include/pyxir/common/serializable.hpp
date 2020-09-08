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

#include "px_stream.hpp"

namespace pyxir {

class ISerializable {

  public:
    ISerializable(){}
    virtual ~ISerializable(){}

    virtual void serialize_px(PxOStringStream &pstream) = 0;
    virtual void deserialize_px(PxIStringStream &pstream) = 0;

    void serialize(std::ostringstream &sstream)
    {
      PxOStringStream pxoss(sstream);
      serialize_px(pxoss);
    }

    void deserialize(std::istringstream &sstream)
    {
      PxIStringStream pxiss(sstream);
      deserialize_px(pxiss);
    }

    template <class T>
    static T &loadFromSStream(std::istringstream &sstream)
    {
      static T self;
      // PxIStringStream pxiss(sstream);
      self.deserialize(sstream);
      return self;
    }
};

} // pyxir 