#pragma once
#include <emptyNN/Layer.hpp>
#include <cassert>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class BatchNormalization: public Layer<Type> {
            protected:
                Type sigma,mu;
            public:
                BatchNormalization(Shape in,Type mu,Type sigma,Activation<Type>* a);
                virtual ~BatchNormalization();
        };        
    } // namespace Layers
} // namespace emptyNN
