#include "emptyNN/layers/Pad.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        Pad<Type>::Pad(Shape in, Shape pad): Layer<Type>(in) {
            // Pad layer is a simple pad of the input
            Shape out = {
                in.width + pad.width,
                in.height + pad.height,
                in.depth + pad.depth
            };
            this->o_shape = out;
            this->o_tensor = new Type[out.size()];
            
        }

        template <class Type>
        void Pad<Type>::forward() {
            // For now, we just copy the tensor to the output
            Type* in = this->i_tensor;
            Type* out = this->o_tensor;
            Shape i_shape = this->i_shape;
            std::copy(in,in+i_shape.size(),out);
        }

        REGISTER_CLASS(Pad,float);

    }
}