#pragma once
#include "emptyNN/Layer.hpp"
     
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class Conv: public Layer<Type> {
            protected:
                Type* filter;
                Shape f_shape;
                ConvParams params;
            public:
                Conv(Shape in, Shape out, ConvParams cp,Activation<Type>* a);
                virtual ~Conv();
                Conv(Shape in, ConvParams cp,Activation<Type>* a);
        };
    }
}