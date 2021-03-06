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
        void SoftmaxFunctor<Type>::operator()(Type* in_tensor, Shape in_shape) {
            Type expProbSum = Type(0.);
            std::transform(in_tensor,in_tensor+in_shape.size(),in_tensor,[&expProbSum](Type el) -> Type {
                expProbSum+=el;
                return (Type)std::exp(double(el));
            });
            std::transform(in_tensor,in_tensor+in_shape.size(),in_tensor,[&expProbSum](Type el) -> Type { return el/expProbSum;});
        }

        template <class Type>
        Type * SoftmaxFunctor<Type>::grad(Type *grad, Shape in_shape) { return grad; };


        REGISTER_CLASS(SoftmaxFunctor,float);
    }
}