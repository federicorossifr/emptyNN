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
#pragma once
#include "emptyNN/layers/Dense.hpp"

namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            class DenseCPUImpl: public Dense<Type> {
                public:
                    DenseCPUImpl(Shape in, Shape out,Activation<Type>* a, bool withBias = true);
                    ~DenseCPUImpl();
                    virtual void forward();
                    virtual Type* backward(Type* grad);

            };
        }
    }
}