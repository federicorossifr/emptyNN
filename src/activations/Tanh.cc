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
#include "emptyNN/activations/Tanh.hpp"
namespace emptyNN {
    namespace Activations {
        template <class Type>
        TanhFunctor<Type>::TanhFunctor() = default;

        template <class Type>
        void TanhFunctor<Type>::operator()(Tensor<Type>& in_tensor) {
            std::transform(in_tensor.begin(),in_tensor.end(),in_tensor.begin(),[](Type& el) -> Type {
                return Type(std::tanh(double(el)));
            });
        }

        template <class Type>
        Tensor<Type> TanhFunctor<Type>::grad(Tensor<Type>& grad) { return grad; };


        REGISTER_CLASS(TanhFunctor,float);
    }
}