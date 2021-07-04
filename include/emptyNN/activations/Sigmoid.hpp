#include "emptyNN/Activation.hpp"

namespace emptyNN {
    namespace Activations {
        template <class Type>
        class Sigmoid: public Activation<Type> {
            public:
                void operator()(Type* tensor);
                void grad()(Type* tensor);
        }
}