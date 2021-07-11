#pragma once
#include "emptyNN/layers/Dense.hpp"

namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            class DenseCPUImpl: public Dense<Type> {
                public:
                    DenseCPUImpl(Shape in, Shape out,Activation<Type>* a);
                    ~DenseCPUImpl();
                    virtual void forward();
                    virtual void backward();

            };
        }
    }
}