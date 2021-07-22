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
#include "emptyNN/activations/Softmax.hpp"
namespace emptyNN {
    namespace Activations {
        template <class Type>
        SoftmaxFunctor<Type>::SoftmaxFunctor() = default;

        template <class Type>
        void SoftmaxFunctor<Type>::operator()(Tensor<Type>& in_tensor) {
            Type expProbSum = Type(0.);
            std::transform(in_tensor.begin(),in_tensor.end(),in_tensor.begin(),[&expProbSum](Type& el) -> Type {
                expProbSum+=el;
                return (Type)std::exp(double(el));
            });
            std::transform(in_tensor.begin(),in_tensor.end(),in_tensor.begin(),[&expProbSum](Type& el) -> Type { return el/expProbSum;});
        }

        template <class Type>
        Tensor<Type> SoftmaxFunctor<Type>::grad(Tensor<Type>& grad) { return grad; };


        REGISTER_CLASS(SoftmaxFunctor,float);
    }
}