#pragma once
#include "emptyNN/Layer.hpp"
     
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class Conv: public Layer<Type> {
            protected:
                Type* filter;
                Type* bias;
                Shape f_shape;
                ConvParams params;
                Shape padding;
            public:
                Conv(Shape in, Shape out, ConvParams cp,Activation<Type>* a);
                virtual ~Conv();
                Conv(Shape in, ConvParams cp,Activation<Type>* a);
                virtual void summary() {Layer<Type>::summary(); std::cout << "Conv weights: " << f_shape.size()*params.kernels + params.kernels << std::endl; }
                virtual std::ostream& operator<<(std::ostream& out);
                virtual std::istream& operator>>(std::istream& ifs);                


        };
    }
}