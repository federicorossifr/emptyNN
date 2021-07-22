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
#include "emptyNN/layers/core/MaxPool_cpu_impl.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            MaxPoolCPUImpl<Type>::MaxPoolCPUImpl(Shape in,PoolParams params,Activation<Type>* a): MaxPooling<Type>(in,params,a){}

            template <class Type>
            MaxPoolCPUImpl<Type>::~MaxPoolCPUImpl() {}

            template <class Type>
            void MaxPoolCPUImpl<Type>::forward() {
                Shape out = this->o_shape;
                Shape in = this->i_shape;
                PoolParams params = this->params;
                Tensor<Type>& i_tensor = this->i_tensor;
                Tensor<Type>& o_tensor = this->o_tensor;
                
                #pragma omp parallel for
                for(size_t depth = 0; depth < in.depth; ++depth) {
                    Type* i_pin = &i_tensor[depth*(in.height*in.width)];
                    Type* o_pin = &i_tensor[depth*(out.height*out.width)];

                    for(size_t i = 0; i < out.height; ++i) {
                        for(size_t j = 0; j < out.width; ++j) {

                            Type max_ = i_pin[0];

                            for(size_t l = 0; l < params.factor.height; ++l ) {
                                for(size_t k = 0; k < params.factor.width; ++k) {
                                    Type pun = i_pin[ (i*params.stride+k)*in.width + l+j*params.stride ];
                                    max_ = std::max(max_, pun);
                                }
                            }

                            o_pin[i*out.width + j] = max_;

                        }
                    }


                }
                
            }



            template <class Type>
            Tensor<Type> MaxPoolCPUImpl<Type>::backward(Tensor<Type>& grad) {return grad;}

            REGISTER_CLASS(MaxPoolCPUImpl,float);
       
        }
    }



}