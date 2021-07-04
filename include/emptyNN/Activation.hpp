#pragma once
#include "emptyNN/Types.hpp"
#include <cmath>
namespace emptyNN {
    template <class Type>
    class Activation {
        public:
            virtual Type operator()(Type el) = 0;
            virtual Type grad(Type el) = 0;
    };
}