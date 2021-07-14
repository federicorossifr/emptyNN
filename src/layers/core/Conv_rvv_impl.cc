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
#include "emptyNN/layers/core/Conv_rvv_impl.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            ConvRVVImpl<Type>::ConvRVVImpl(Shape in, Shape out, ConvParams cp,Activation<Type>* a): Conv<Type>(in,out,cp,a){}

            template <class Type>
            ConvRVVImpl<Type>::ConvRVVImpl(Shape in, ConvParams cp,Activation<Type>* a): Conv<Type>(in,cp,a){}

            template <class Type>
            ConvRVVImpl<Type>::~ConvRVVImpl() {}

            template <class Type>
            void ConvRVVImpl<Type>::forward() {
                Shape out = this->o_shape;
                Shape in = this->i_shape;
                Shape fil = this->f_shape;
                Shape padding = this->padding;
                Type* filter = this->filter;
                Type* i_tensor = this->i_tensor;
                Type* o_tensor = this->o_tensor;
                size_t stride = this->params.stride;
                size_t kernels = this->params.kernels;
                #pragma omp parallel for                
                for(size_t kernel = 0; kernel < kernels; ++kernel) {
                    Type* o_pin = &o_tensor[kernel*(out.height*out.width)];

                    for(size_t i = 0; i < out.height; ++i) {
                        for(size_t j = 0; j < out.width; ++j) {
                            Type daccum(0);


                            for(size_t depth = 0; depth < fil.depth; ++depth) {

                                Type* i_pin = &i_tensor[depth*(in.height*in.width)];
                                
                                Type* f_pin = &filter[depth*(fil.height*fil.width)];
                                Type accum(0);

                                for(size_t k = 0; k < fil.height; ++k) {
                                    for(size_t l = 0; l < fil.width; ++l) {
                                        Type _a = f_pin[ k*fil.width + l ];
                                        Type _b = i_pin[ (i*stride+k)*in.width + l+j*stride ];
                                        accum +=  _a * _b;
                                    }
                                }
                                daccum+=accum;
                                
                            }
                            o_pin[i*out.width+j] = daccum;

                        }
                    }
                    //delete [] i_padded;                    
                }
            }

            template <class Type>
            void ConvRVVImpl<Type>::backward(Type* grad) {}            
                

            REGISTER_CLASS(ConvRVVImpl,float)

        }
    }



}