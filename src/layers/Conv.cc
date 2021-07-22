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
#include "emptyNN/layers/Conv.hpp"
#include <emptyNN/utils/tensor_utils.hpp>
#include <iostream>
namespace emptyNN {
    namespace Layers {

        template <class Type>
        Conv<Type>::~Conv() {

        }

        template <class Type>
        Conv<Type>::Conv(Shape in, ConvParams cp,Activation<Type>* a, bool withBias): Layer<Type>(in,a),f_shape(cp.filter),params(cp),hasBias(withBias) {
            size_t filter_size = cp.filter.height*cp.filter.width*cp.filter.depth;
            filter_size*= (cp.isDepthWise)? 1:cp.kernels;
            filter = Tensor<Type>(filter_size);


            emptyNN::Utils::Tensors::fillRandomUniform<Type>(filter,filter_size);
            if (hasBias) {
                bias = Tensor<Type>(cp.kernels);
                emptyNN::Utils::Tensors::fillRandomUniform<Type>(bias,cp.kernels);
            }


            size_t des_o_width = (cp.padding == PaddingType::SAME)? in.width : ceil(float(in.width - cp.filter.width+1)/cp.stride);
            size_t des_o_height = (cp.padding == PaddingType::SAME)? in.height : ceil(float(in.height - cp.filter.height+1)/cp.stride);
            
            Shape req_padding = {
                in.width*(cp.stride - 1)+cp.filter.width - 1,
                in.height*(cp.stride - 1)+cp.filter.height - 1
            };

            if (cp.padding == PaddingType::SAME) {
                this->padding = req_padding;

                // Expand the input tensor buffer to the padded dimension
                this->i_tensor.resize( ( in.width + req_padding.width )*( in.height + req_padding.height )*in.depth );
            }
            else this->padding = {0,0,0};
            Shape out = {des_o_width,
                         des_o_height,
                         cp.kernels};

            this->o_tensor = Tensor<Type>(out.width*out.height*out.depth);
            this->o_shape = out;
        }        

        template <class Type>
        std::ostream& Conv<Type>::operator<<(std::ostream& out) {
            size_t filter_size = params.filter.height*params.filter.width*params.filter.depth;
            filter_size*= (params.isDepthWise)? 1:params.kernels;
            out.write(reinterpret_cast<char*>(filter.data()),filter_size*sizeof(Type));
            if (hasBias)
                out.write(reinterpret_cast<char*>(bias.data()),params.kernels*sizeof(Type));
            return out;
        }

        template <class Type>
        std::istream& Conv<Type>::operator>>(std::istream& ifs) {
            size_t filter_size = params.filter.height*params.filter.width*params.filter.depth;
            filter_size*= (params.isDepthWise)? 1:params.kernels;            
            ifs.read(reinterpret_cast<char*>(filter.data()),filter_size*sizeof(Type));
            if (hasBias)
                ifs.read(reinterpret_cast<char*>(bias.data()),params.kernels*sizeof(Type));
            return ifs;
        }        
        REGISTER_CLASS(Conv,float)

    }


    
}