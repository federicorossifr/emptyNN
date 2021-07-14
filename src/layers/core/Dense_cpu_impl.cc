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
#include "emptyNN/layers/core/Dense_cpu_impl.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            DenseCPUImpl<Type>::DenseCPUImpl(Shape in, Shape out,Activation<Type>* a): Dense<Type>(in,out,a){}

            template <class Type>
            DenseCPUImpl<Type>::~DenseCPUImpl() {}

            template <class Type>
            void DenseCPUImpl<Type>::forward() {
                Shape out = this->o_shape;
                Shape in = this->i_shape;
                Type* i_tensor = this->i_tensor;
                Type* o_tensor = this->o_tensor;
                Type* fc       = this->connections;
                size_t in_rows = in.size(),
                       out_rows = out.size();
                for(size_t i = 0; i < out_rows; ++i) {
                    Type accum = Type(0.f);

                    for(size_t j = 0; j < in_rows ; ++j) {
                        
                        accum += i_tensor[j]*fc[j+i*in_rows];
                    }

                    o_tensor[i] = accum;
                }

            }

 
            template <class Type>
            void DenseCPUImpl<Type>::backward(Type* dy) {
                // Grad should have the same elements as the layer output
                // dW is the outer product between grad and the layer input
                // If the Dense layer has 
                // input (X,1,1) 
                // (X,Y) weights
                // output (Y,1,1)
                // The grad will be for sure (Y,1,1)
                // So the outer product between (Y,1) and (X,1)^ 
                Shape out = this->o_shape;
                Shape in =  this->i_shape;
                Type* x = this->i_tensor;
                Type* W = this->connections;
                size_t dy_rows = out.size();
                size_t x_rows = in.size();
                // Weight derivative 
                Type* dw = new Type[out.size()*in.size()];

                for(size_t i = 0; i < dy_rows; ++i) {
                    for(size_t j = 0; j < x_rows; ++j) {
                        dw[i*x_rows+j] = dy[i]*x[j];
                    }
                }

                Type* dx = new Type[in.size()];

                for(size_t i = 0; i < x_rows; ++i) {
                    
                    for(size_t j = 0; j < dy_rows; ++j) {
                        dx[i] = dy[j] * W[j*x_rows+i];
                    }
                }

                //return dx;

            }            

            REGISTER_CLASS(DenseCPUImpl,float);
       
        }
    }



}