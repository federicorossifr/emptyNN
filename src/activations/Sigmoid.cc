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
#include "emptyNN/activations/Sigmoid.hpp"
namespace emptyNN {
    namespace Activations {
        template <class Type>
        SigmoidFunctor<Type>::SigmoidFunctor(){};

        template <class Type>
        void SigmoidFunctor<Type>::operator()(Type* in_tensor, Shape in_shape) {
            std::transform(in_tensor,in_tensor+in_shape.size(),in_tensor,[](Type el) -> Type {
                return Type(1.)/(Type(1.)+(Type)std::exp((double)-el));
            });
        }

        template <class Type>
        Type * SigmoidFunctor<Type>::grad(Type *grad, Shape in_shape) {
            auto sigmoid = [](Type x) -> Type {return Type(1.)/(Type(1.)+(Type)std::exp((double)-x));};
            std::transform(grad,grad+in_shape.size(),grad,[&sigmoid](Type x) {
                return x* sigmoid(x) * ( Type(1.) - sigmoid(x) );
            });
            return grad;
        };


        REGISTER_CLASS(SigmoidFunctor,float);
    }
}