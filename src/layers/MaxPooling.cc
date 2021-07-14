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
#include "emptyNN/layers/MaxPooling.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        MaxPooling<Type>::MaxPooling(Shape in, PoolParams params,Activation<Type>* a): Layer<Type>(in,a),params(params) {
            Shape out = {
                (size_t)ceil( double( double(in.width-params.factor.width+1)/params.stride ) ),
                (size_t)ceil( double( double(in.height-params.factor.height+1)/params.stride ) ),
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