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
#include "emptyNN/layers/core/BatchNorm_cpu_impl.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            BatchNormCPUImpl<Type>::BatchNormCPUImpl(Shape in, Type mu, Type sigma, std::unique_ptr<Activation<Type>> a): BatchNormalization<Type>(in,mu,sigma,std::move(a)){}

            template <class Type>
            BatchNormCPUImpl<Type>::~BatchNormCPUImpl() {}

            template <class Type>
            void BatchNormCPUImpl<Type>::forward() {
                Type mu = this->mu;
                Type sigma = this->sigma;
                Tensor<Type>& i_tensor = this->i_tensor;
                Tensor<Type>& o_tensor = this->o_tensor;
                size_t in_size = this->i_shape.size();

                #pragma omp parallel for simd
                for(int i = 0; i < in_size; ++i)
                    o_tensor[i] = (Type(i_tensor[i])-Type(mu))/Type(sigma);
            }



            template <class Type>
            Tensor<Type> BatchNormCPUImpl<Type>::backward(Tensor<Type>& grad) {return grad;}

            REGISTER_CLASS(BatchNormCPUImpl,float);
       
        }

    }



}