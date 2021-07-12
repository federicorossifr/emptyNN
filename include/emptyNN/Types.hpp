#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <omp.h>
#include <exception>

#ifdef USE_POSIT
#include <posit.h>
#include <anyfloat.hpp>
#endif

namespace emptyNN {
    typedef struct Shape {
        size_t width;
        size_t height;
        size_t depth;

        size_t size() {return width*height*depth;};
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
        virtual const char* what() const throw();

        public:
            DeviceNotAllowed(Device d);

    };    

    #ifdef USE_POSIT
    #define REGISTER_CLASS(CLASS,TYPE) \
        template class CLASS<float>; \
        template class CLASS<P16_1>; \
        template class CLASS<P16_0>; \
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
        using Posit8_0 = P8fx;
        using Bfloat16 = binary16alt_emu;
        using Bfloat8 = binary8_emu;

    #endif
}