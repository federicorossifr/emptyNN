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
#include <algorithm>
#include "emptyNN/activations/ReLU.hpp"
namespace emptyNN {
    namespace Activations {
        template <class Type>
        ReLuFunctor<Type>::ReLuFunctor()= default;

        template <class Type>
        void ReLuFunctor<Type>::operator()(Type* in_tensor, Shape in_shape) {
            std::transform(in_tensor,in_tensor+in_shape.size(),in_tensor,[](Type el) -> Type {
                return (el >= Type(0.f))? el : Type(1.);
            });
        }

        template <class Type>
        Type ReLuFunctor<Type>::grad(Type el) { return el; };


        REGISTER_CLASS(ReLuFunctor,float);
    }
}