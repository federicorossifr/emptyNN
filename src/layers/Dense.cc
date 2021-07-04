#include "emptyNN/layers/Dense.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        Dense<Type>::Dense(Shape in, Shape out): Layer<Type>(in,out) {
            assert(in.depth == 1 && out.depth == 1);
            // In-vector   1 x (in.height * in.width) 
            // Out-vector  (out.height * out.width) x 1
            // Connection matrix (in.height * in.width) x (out.height * out.width)
            connections = new Type[in.height * in.width * out.height * out.width];
        }

        template <class Type>
        Dense<Type>::~Dense() {
            delete[] connections;
        }    

        template class Dense<float>;

    }
}