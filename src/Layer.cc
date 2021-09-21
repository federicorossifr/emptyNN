/*
emptyNN
Copyright (C) 2021 Federico Rossi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "emptyNN/Layer.hpp"
#include <iostream>

namespace emptyNN {

    template <class Type>
    Layer<Type>::Layer(Shape in,Shape out,std::unique_ptr<Activation<Type>> a): i_shape(in),o_shape(out),activation(std::move(a)) {
        i_tensor = Tensor<Type>(in.width*in.height*in.depth);
        o_tensor = Tensor<Type>(out.width*out.height*out.depth);
    }

    template <class Type>
    Layer<Type>::~Layer() {}

    template <class Type>
    Layer<Type>::Layer(Shape in,std::unique_ptr<Activation<Type>> a): i_shape(in),activation(std::move(a)) {
        i_tensor = Tensor<Type>(in.width*in.height*in.depth);
    
    }

    template <class Type>
    void Layer<Type>::fillInTensor(Tensor<Type>& in) {
        std::copy(in.begin(),in.end(),i_tensor.begin());
    }

    template <class Type>
    Tensor<Type>& Layer<Type>::operator()() {
       forward();
       activate();
       return o_tensor;
    }    

    template <class Type>
    Shape Layer<Type>::getInputShape() {
        return i_shape;
    }

    template <class Type>
    void Layer<Type>::summary() {
        std::cout << this << " In: (" << getInputShape().width << ", " << getInputShape().height << "," << getInputShape().depth << ")" 
                          << " Out: (" << getOutputShape().width << ", " << getOutputShape().height << "," << getOutputShape().depth << ")" <<std::endl;

    }    

    template <class Type>
    Shape Layer<Type>::getOutputShape() {
        return o_shape;
    }    

    template <class Type>
    void Layer<Type>::activate() {
        auto* a = this->activation.get();
        Tensor<Type>& o_tensor = this->o_tensor;
        if( a == nullptr) return;

        (*a)(o_tensor);
    }           
    /* Type specific implementations */
    REGISTER_CLASS(Layer,float)

}