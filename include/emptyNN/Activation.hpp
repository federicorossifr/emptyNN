#pragma once
#include "emptyNN/Types.hpp"
#include <cmath>
namespace emptyNN {
    template <class Type>
    class Activation {
        public:
            virtual void operator()(Type* in_tensor, Shape in_shape) = 0;
            virtual Type grad(Type el) = 0;
            virtual ~Activation() {};
    };
}