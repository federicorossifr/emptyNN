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