#include "emptyNN/layers/Conv.hpp"

namespace emptyNN {
    namespace Layers {
        template <class Type>
        Conv<Type>::Conv(Shape in, Shape out, ConvParams cp,Activation<Type>* a): Layer<Type>(in,out,a),f_shape(cp.filter),params(cp) {
            filter = new Type[cp.filter.height*cp.filter.width*cp.kernels];
        }

        template <class Type>
        Conv<Type>::Conv(Shape in, ConvParams cp,Activation<Type>* a): Layer<Type>(in,a),f_shape(cp.filter),params(cp) {
            filter = new Type[cp.filter.height*cp.filter.width*cp.filter.depth*cp.kernels];
            Shape out = {in.width - cp.filter.width+1/cp.stride,
                         in.height - cp.filter.height+1/cp.stride,
                         cp.kernels};
            this->o_tensor = new Type[out.width*out.height*out.depth];
            this->o_shape = out;
        }        
    }


    REGISTER_CONV(float)
}