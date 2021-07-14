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
#include "emptyNN/Layer.hpp"
     
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class MaxPooling: public Layer<Type> {
            protected:
                Type* filter;
                Shape f_shape;
                PoolParams params;
            public:
                MaxPooling(Shape in, Shape out, PoolParams cp,Activation<Type>* a);
                virtual ~MaxPooling();
                MaxPooling(Shape in, PoolParams cp,Activation<Type>* a);
                virtual std::ostream& operator<<(std::ostream& out)  {return out;}
                virtual std::istream& operator>>(std::istream& ifs)  {return ifs;}                

        };
    }
}