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
#include "emptyNN/layers/Dense.hpp"
#include <emptyNN/utils/tensor_utils.hpp>

#include <iostream>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        Dense<Type>::Dense(Shape in, Shape out,std::unique_ptr<Activation<Type>> a, bool withBias): Layer<Type>(in, out, std::move(a)),hasBias(withBias) {
            // assert(in.depth == 1 && out.depth == 1);
            // In-vector   1 x (in.height * in.width) 
            // Out-vector  (out.height * out.width) x 1
            // Connection matrix (in.height * in.width) x (out.height * out.width)
            size_t conn_size = in.size() * out.size();
            connections = Tensor<Type>(conn_size);
            emptyNN::Utils::Tensors::fillRandomUniform<Type>(connections,conn_size);

            if(hasBias) {
                bias = Tensor<Type>(out.size());
                emptyNN::Utils::Tensors::fillRandomUniform<Type>(bias,out.size());
            }

        }

        template <class Type>
        Dense<Type>::~Dense() {;
        }    

        template <class Type>
        std::ostream& Dense<Type>::operator<<(std::ostream& ofs) {
            Shape in = this->i_shape;
            Shape out = this->o_shape;
            size_t connection_size = in.size() * out.size();
            ofs.write(reinterpret_cast<char*>(connections.data()),connection_size*sizeof(Type))
               .write(reinterpret_cast<char*>(bias.data()),out.size()*sizeof(Type));
            return ofs;
        }        

        template <class Type>
        std::istream& Dense<Type>::operator>>(std::istream& ifs) {
            Shape in = this->i_shape;
            Shape out = this->o_shape;
            size_t connection_size = in.size() * out.size();
            ifs.read(reinterpret_cast<char*>(connections.data()),connection_size*sizeof(Type))
                .read(reinterpret_cast<char*>(bias.data()),out.size()*sizeof(Type));
            return ifs;
        }              

        REGISTER_CLASS(Dense,float);

    }
}