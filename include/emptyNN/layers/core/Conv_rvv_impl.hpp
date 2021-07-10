#pragma once
#include "emptyNN/layers/Conv.hpp"

namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            class ConvRVVImpl: public Conv<Type> {
                public:
                    ConvRVVImpl(Shape in, Shape out, ConvParams cp,Activation<Type>* a = nullptr);
                    ~ConvRVVImpl();
                    ConvRVVImpl(Shape in, ConvParams cp,Activation<Type>* a = nullptr);
                    virtual void forward();
                    virtual void backward();
                    virtual void activate();

            };
        }
    }
}