#include "emptyNN/layers/Dense.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        Dense<Type>::Dense(Shape in, Shape out,Activation<Type>* a): Layer<Type>(in,out,a) {
            // assert(in.depth == 1 && out.depth == 1);
            // In-vector   1 x (in.height * in.width) 
            // Out-vector  (out.height * out.width) x 1
            // Connection matrix (in.height * in.width) x (out.height * out.width)
            connections = new Type[in.height * in.width * out.height * out.width * in.depth * out.depth];
        }

        template <class Type>
        Dense<Type>::~Dense() {
            delete[] connections;
        }    

        REGISTER_CLASS(Dense,float);

    }
}