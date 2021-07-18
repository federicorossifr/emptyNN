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
#include "emptyNN/layers/core/Conv_cpu_impl.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        namespace Impl {

            template <class Type>
            ConvCPUImpl<Type>::ConvCPUImpl(Shape in, ConvParams cp,Activation<Type>* a, bool hasBias): Conv<Type>(in,cp,a,hasBias){}

            template <class Type>
            ConvCPUImpl<Type>::~ConvCPUImpl() {}

            template <class Type>
            void ConvCPUImpl<Type>::forward() {
                Shape out = this->o_shape;
                Shape in = this->i_shape;
                Shape fil = this->f_shape;
                Type* filter = this->filter;
                Type* i_tensor = this->i_tensor;
                Type* o_tensor = this->o_tensor;
                size_t stride = this->params.stride;
                size_t kernels = this->params.kernels;
                #pragma omp parallel for                
                for(size_t kernel = 0; kernel < kernels; ++kernel) {
                    Type* o_pin = &o_tensor[kernel*(out.height*out.width)];
                    Type* f_pin = &filter[kernel*(fil.height*fil.width*fil.depth)];

                    for(size_t i = 0; i < out.height; ++i) {
                        for(size_t j = 0; j < out.width; ++j) {
                            Type daccum(0);


                            for(size_t depth = 0; depth < fil.depth; ++depth) {

                                Type* i_pin = &i_tensor[depth*(in.height*in.width)];
                                
                                Type* ff_pin = &f_pin[depth*(fil.height*fil.width)];
                                Type accum(0);

                                for(size_t k = 0; k < fil.height; ++k) {
                                    for(size_t l = 0; l < fil.width; ++l) {
                                        Type _a = ff_pin[ k*fil.width + l ];
                                        Type _b = i_pin[ (i*stride+k)*in.width + l+j*stride ];
                                        accum +=  _a * _b;
                                    }
                                }
                                daccum+=accum;
                                
                            }
                            o_pin[i*out.width+j] = daccum;

                        }
                    }
                }
            }




            template <class Type>
            Type* ConvCPUImpl<Type>::backward(Type* grad) {
                Shape out = this->o_shape;
                Shape in  = this->i_shape;
                Shape fi  = this->f_shape;
                Type* xp = this->i_tensor;
                Type* f  = this->filter;
                size_t kernels = this->params.kernels;
                size_t stride = this->params.stride;

                // Reserve space for the weight gradient
                Type* dW = new Type[fi.size()*kernels];
                std::fill(dW,dW+fi.size()*kernels,Type(0.));

                //Reserve space for the backpropagation gradient
                Type* dxp = new Type[in.size()];
                std::fill(dxp,dxp+in.size(),Type(0.));

                // Compute dW
                for(size_t kernel{}; kernel < kernels; ++kernel) {
                    // Pin the current gradient kernel pointer
                    Type* o_pin = &grad[kernel*(out.height*out.width)];

                    // Pin the current weight kernel pointer
                    Type* dw_pin = &dW[kernel*(fi.height*fi.width*fi.depth)];

                    // Iterate on all gradient elements from next layer
                    for(size_t i{}; i < out.height; ++i) {
                        for (size_t j{}; j < out.width; ++j) {

                            // Pin the gradient value to be multiplied with the filter window
                            Type _o = o_pin[i*out.width + j];

                            // Iterate on all weight elements to be computed
                            for(size_t k{}; k < fi.height; ++k) {
                                for(size_t l{}; l < fi.height; ++l) {
                                    for(size_t d{}; d < fi.depth; ++d) {

                                        // Pin the channel for current weight kernel
                                        Type* ddw_pin = &dw_pin[d*(fi.height*fi.width)];

                                        // Pin channel for current input
                                        Type* i_pin = &xp[d*(in.height*in.width)];

                                        Type _a = i_pin[ (i*stride+k)*in.width + l+j*stride ];
                                        ddw_pin[k*fi.width + l] += _a*_o;
                                    }
                                }
                            }
                        }
                    }
                }

                // Compute dx
                // Iterate on all elements of out gradient
                for(size_t kernel{}; kernel < kernels; ++kernel) {
                    // Pin the current gradient kernel pointer
                    Type* o_pin = &grad[kernel*(out.height*out.width)];

                    // Pin the current weight kernel pointer
                    Type* f_pin = &f[kernel*(fi.height*fi.width*fi.depth)];

                    // Iterate on all gradient elements from next layer
                    for(size_t i{}; i < out.height; ++i) {
                        for (size_t j{}; j < out.width; ++j) {

                            // Pin the gradient value to be multiplied with the filter window
                            Type _o = o_pin[i*out.width + j];

                            // Iterate on all weight elements to be computed
                            for(size_t k{}; k < fi.height; ++k) {
                                for(size_t l{}; l < fi.height; ++l) {
                                    for(size_t d{}; d < fi.depth; ++d) {

                                        // Pin the channel for current weight kernel
                                        Type* df_pin = &f_pin[d*(fi.height*fi.width)];

                                        // Pin channel for current back gradient
                                        Type* dxp_pin = &dxp[d*(in.height*in.width)];

                                        Type& _a = dxp_pin[ (i*stride+k)*in.width + l+j*stride ];
                                        Type& _f = df_pin[k*fi.width + l];
                                        _a = _f*_o;
                                    }
                                }
                            }
                        }
                    }
                }

                return dxp;
            }
                

            REGISTER_CLASS(ConvCPUImpl,float)

        }
    }



}