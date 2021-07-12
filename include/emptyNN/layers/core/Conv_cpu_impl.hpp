#pragma once
#include "emptyNN/layers/Conv.hpp"
#include <memory>
namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            class ConvCPUImpl: public Conv<Type> {
                public:
                    ConvCPUImpl(Shape in, Shape out, ConvParams cp,Activation<Type>* a = nullptr);
                    ~ConvCPUImpl();
                    ConvCPUImpl(Shape in, ConvParams cp,Activation<Type>* a = nullptr);
                    virtual void forward();
                    virtual void backward();

            };
        }
    }
}