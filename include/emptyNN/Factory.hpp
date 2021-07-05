#pragma once
#include "emptyNN/layers/core/Conv_cpu_impl.hpp"
#include "emptyNN/layers/core/Dense_cpu_impl.hpp"
#include "emptyNN/layers/core/BatchNorm_cpu_impl.hpp"
#include "emptyNN/layers/core/MaxPool_cpu_impl.hpp"
#include "emptyNN/activations/Elu.hpp"

namespace emptyNN {
    namespace Factory {
        namespace Layers {
            
            template <class Type>
            Layer<Type>* Convolution(Shape in,ConvParams params,Activation<Type>* a,Device device);

            template <class Type>
            Layer<Type>* Dense(Shape in,Shape out,Activation<Type>* a,Device device);

            template <class Type>
            Layer<Type>* BatchNorm(Shape in, Type mu, Type sigma, Activation<Type>* a, Device device);

            template <class Type>
            Layer<Type>* MaxPool(Shape in, PoolParams params, Activation<Type>* a, Device device);

        }

        namespace Activations {
            template <class Type>
            Activation<Type>* Elu(Type alpha = 1);
        }
    }
}