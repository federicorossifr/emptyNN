#pragma once
#include "emptyNN/layers/Conv.hpp"

namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            class DWConvCPUImpl: public Conv<Type> {
                public:
                    DWConvCPUImpl(Shape in, Shape out, ConvParams cp,Activation<Type>* a = nullptr);
                    ~DWConvCPUImpl();
                    DWConvCPUImpl(Shape in, ConvParams cp,Activation<Type>* a = nullptr);
                    virtual void forward();
                    virtual void backward();

            };
        }
    }
}