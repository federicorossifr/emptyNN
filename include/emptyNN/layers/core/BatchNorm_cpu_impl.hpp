#pragma once
#include "emptyNN/layers/BatchNormalization.hpp"

namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            class BatchNormCPUImpl: public BatchNormalization<Type> {
                public:
                    BatchNormCPUImpl(Shape in, Type mu, Type sigma, Activation<Type>* a);
                    ~BatchNormCPUImpl();
                    virtual void forward();
                    virtual void backward();

            };
        }
    }
}