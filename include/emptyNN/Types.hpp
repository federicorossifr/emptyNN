#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <exception>
namespace emptyNN {
    typedef struct {
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
    } ConvParams;

    typedef struct  {
        Shape factor;
        size_t stride;
    } PoolParams;    

    typedef enum {
        CPU,CPU_RVV,CPU_SVE,GPU
    } Device;


    class DeviceNotAllowed: public std::exception {
        Device d;
        virtual const char* what() const throw();

        public:
            DeviceNotAllowed(Device d);

    };    

    #define REGISTER_LAYER_TYPE(TYPE) template class Layer<TYPE>;
    #define REGISTER_CONV(TYPE) template class Layers::Conv<TYPE>;
    #define REGISTER_CONV_CPU_IMPL(TYPE) template class Layers::Impl::ConvCPUImpl<TYPE>;

}