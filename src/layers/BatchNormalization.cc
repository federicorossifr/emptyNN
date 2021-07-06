#include "emptyNN/layers/BatchNormalization.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        BatchNormalization<Type>::BatchNormalization(Shape in, Type mu,Type sigma,Activation<Type>* a): Layer<Type>(in,in,a),sigma(sigma),mu(mu) {}

        template <class Type>
        BatchNormalization<Type>::~BatchNormalization() {}    

        REGISTER_CLASS(BatchNormalization,float);

    }
}