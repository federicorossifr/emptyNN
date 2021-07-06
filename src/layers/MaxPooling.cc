#include "emptyNN/layers/MaxPooling.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        MaxPooling<Type>::MaxPooling(Shape in, PoolParams params,Activation<Type>* a): Layer<Type>(in,a),params(params) {
            Shape out = {
                (size_t)floor( double( (in.width-params.factor.width+1)/params.stride ) ),
                (size_t)floor( double( (in.height-params.factor.height+1)/params.stride ) ),
                in.depth
            };
            this->o_shape = out;
            this->o_tensor = new Type[out.width*out.height*out.depth];

        }

        template <class Type>
        MaxPooling<Type>::~MaxPooling() {}    

        REGISTER_CLASS(MaxPooling,float);

    }
}