#pragma once
#include "emptyNN/Layer.hpp"
     
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class MaxPooling: public Layer<Type> {
            protected:
                Type* filter;
                Shape f_shape;
                PoolParams params;
            public:
                MaxPooling(Shape in, Shape out, PoolParams cp,Activation<Type>* a);
                virtual ~MaxPooling();
                MaxPooling(Shape in, PoolParams cp,Activation<Type>* a);
                virtual std::ostream& operator<<(std::ostream& out)  {return out;}
        };
    }
}