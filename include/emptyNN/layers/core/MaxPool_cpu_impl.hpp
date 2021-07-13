#pragma once
#include "emptyNN/layers/MaxPooling.hpp"

namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            class MaxPoolCPUImpl: public MaxPooling<Type> {
                public:
                    MaxPoolCPUImpl(Shape in,PoolParams params,Activation<Type>* a);
                    ~MaxPoolCPUImpl();
                    virtual void forward();
                    virtual void backward(Type* grad);

            };
        }
    }
}