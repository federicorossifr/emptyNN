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
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <omp.h>
#include <exception>
#include <iostream>
#ifdef USE_POSIT
#include <posit.h>
#include <anyfloat.hpp>
#endif

namespace emptyNN {
    namespace Random {
        extern std::uint32_t globalSeed;
    }
    typedef struct Shape {
        size_t width;
        size_t height;
        size_t depth;

        size_t size() const {return width*height*depth;};
    } Shape;
    bool operator==(Shape a,Shape b);

    typedef enum {
        ZERO,SAME,NONE
    } PaddingType;

    typedef struct  {
        Shape filter;
        size_t kernels;
        size_t stride;
        PaddingType padding;
        bool isDepthWise;
    } ConvParams;

    typedef struct  {
        Shape factor;
        size_t stride;
    } PoolParams;    

    typedef enum {
        CPU,CPU_RVV,CPU_SVE,GPU
    } Device;

    typedef struct {
        bool halve;
        size_t block_size;
        bool identity;
    } ResBlockParams;


    class DeviceNotAllowed: public std::exception {
        Device d;
        const char* what() const throw() override;

        public:
            explicit DeviceNotAllowed(Device d);

    };    

    #ifdef USE_POSIT
    #define REGISTER_CLASS(CLASS,TYPE) \
        template class CLASS<float>; \
        template class CLASS<Posit16_1>; \
        template class CLASS<Posit16_0>; \
        template class CLASS<Posit8_0>; \
        template class CLASS<Bfloat16>; \
        template class CLASS<Bfloat8>; \
        template class CLASS<FloatEmu>;
    #else
    #define REGISTER_CLASS(CLASS,TYPE) \
        template class CLASS<float>;
    #endif

    #ifdef USE_POSIT
        using FloatEmu = binary32_emu;
        using Posit16_1 = P16_1;
        using Posit16_0 = P16_0;
        using Posit8_0 = P8;
        using Bfloat16 = binary16alt_emu;
        using Bfloat8 = binary8_emu;

    #endif


}