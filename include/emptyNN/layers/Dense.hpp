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
#include <emptyNN/Layer.hpp>
#include <cassert>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class Dense: public Layer<Type> {
            protected:
                Type* connections;
                Type* bias;
                const bool hasBias;
            public:
                Dense(Shape in, Shape out,Activation<Type>* a, bool withBias = true);
                virtual ~Dense();
                virtual void summary() {
                    Layer<Type>::summary(); 
                    std::cout << "Dense weights: " << this->i_shape.size() * this->o_shape.size() + this->o_shape.size() << std::endl; 
                }

                virtual std::ostream& operator<<(std::ostream& out);
                virtual std::istream& operator>>(std::istream& ifs);                

        };        
    } // namespace Layers
} // namespace emptyNN
