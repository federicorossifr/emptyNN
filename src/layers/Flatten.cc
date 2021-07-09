#include "emptyNN/layers/Flatten.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        Flatten<Type>::Flatten(Shape in): Layer<Type>(in) {
            // Flatten layer is just a simple pass-through layer 
            // that reshapes a tensor into a monodimensional one
            Shape out = {
                in.size(),
                1,
                1
            };
            this->o_shape = out;
            this->o_tensor = new Type[out.size()];
        }

        template <class Type>
        void Flatten<Type>::forward() {
            // Flattening is just a copy from input tensor to output tensor
            Type* in = this->i_tensor;
            Type* out = this->o_tensor;
            Shape i_shape = this->i_shape;
            std::copy(in,in+i_shape.size(),out);
        }

        REGISTER_CLASS(Flatten,float);

    }
}