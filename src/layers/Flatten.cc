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
            this->o_tensor = Tensor<Type>(out.size());
        }

        template <class Type>
        void Flatten<Type>::forward() {
            // Flattening is just a copy from input tensor to output tensor
            Tensor<Type>& in = this->i_tensor;
            Tensor<Type>&  out = this->o_tensor;
            Shape i_shape = this->i_shape;
            std::copy(in.begin(),in.end(),out.begin());
        }

        REGISTER_CLASS(Flatten,float);

    }
}