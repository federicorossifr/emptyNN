#pragma once
#include "emptyNN/Activation.hpp"

namespace emptyNN {
    namespace Activations {
        template <class Type>
        class EluFunctor: public Activation<Type> {
            private:
                Type alpha;
            public:
                EluFunctor(Type alpha=1.);
                virtual void operator()(Type* in_tensor, Shape in_shape);
                Type grad(Type el); 
                virtual ~EluFunctor() {};

        };
    }
}