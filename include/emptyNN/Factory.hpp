#pragma once
#include "emptyNN/layers/core/Conv_cpu_impl.hpp"
#include "emptyNN/activations/Elu.hpp"

namespace emptyNN {
    namespace Factory {
        namespace Layers {
            
            template <class Type>
            Layer<Type>* Convolution(Shape in,ConvParams params,Activation<Type>* a,Device device);


        }

        namespace Activations {
            template <class Type>
            Activation<Type>* Elu(Type alpha = 1);
        }
    }
}