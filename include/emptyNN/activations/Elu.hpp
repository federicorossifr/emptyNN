#pragma once
#include "emptyNN/Activation.hpp"

namespace emptyNN {
    namespace Activations {
        template <class Type>
        class EluFunctor: public Activation<Type> {
            private:
                Type alpha;
            public:
                EluFunctor(Type alpha=1);
                Type operator()(Type el); 
                Type grad(Type el); 
        };
    }
}