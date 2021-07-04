#include "emptyNN/Layer.hpp"
#include <iostream>

namespace emptyNN {

    template <class Type>
    Layer<Type>::Layer(Shape in,Shape out,Activation<Type>* a): i_shape(in),o_shape(out),activation(a) {
        i_tensor = new Type[in.width*in.height*in.depth];
        o_tensor = new Type[out.width*out.height*out.depth];
    }

    template <class Type>
    Layer<Type>::~Layer() {
        delete[] i_tensor;
        delete[] o_tensor;
        if(activation != nullptr)
            delete activation;
    }

    template <class Type>
    Layer<Type>::Layer(Shape in,Activation<Type>* a): i_shape(in),activation(a) {
        i_tensor = new Type[in.width*in.height*in.depth];
    }

    template <class Type>
    void Layer<Type>::fillInTensor(Type* in) {
        std::memcpy(i_tensor,in,i_shape.size());
    }

    template <class Type>
    Type* Layer<Type>::operator()() {
       forward();
       activate();
       return o_tensor;
    }    

    template <class Type>
    Shape Layer<Type>::getInputShape() {
        return i_shape;
    }

    template <class Type>
    Shape Layer<Type>::getOutputShape() {
        return o_shape;
    }    
    /* Type specific implementations */
    REGISTER_LAYER_TYPE(float)

}