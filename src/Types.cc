/*
emptyNN
Copyright (C) 2021 Federico Rossi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "emptyNN/Types.hpp"
namespace emptyNN {

    namespace Random
    {
        std::uint32_t globalSeed(1);
    } // namespace Random
    
    std::string DeviceToStr_(Device d) {
        switch (d)
            {
                case CPU: return "CPU";
                case CPU_RVV: return "CPU_RVV";
                case CPU_SVE: return "CPU_SVE";
                case GPU: return "GPU";
                default: return "INVALID";
            }
    }


    DeviceNotAllowed::DeviceNotAllowed(Device d): d(d){};
    const char* DeviceNotAllowed::what() const throw() {
        std::string* msg = new std::string("Device: ");
        *msg += DeviceToStr_(d) + " not allowed";
        return msg->c_str();
    }

    bool operator==(Shape a,Shape b) {
        return (a.depth == b.depth) &
               (a.width == b.width) &
               (a.height == b.height);
    }
}    



