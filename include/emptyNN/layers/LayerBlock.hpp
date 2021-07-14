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
#include <emptyNN/Sequential.hpp>
#include <emptyNN/Layer.hpp>
#include <emptyNN/Types.hpp>
#include <vector>
#include <cassert>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class LayerBlock: public Layer<Type> {
            protected:
                std::vector<std::vector<Layer<Type>*>> block;
            public:
                LayerBlock(Shape in, Shape out);   
                LayerBlock(Shape in, Shape out,std::vector<std::vector<Layer<Type>*>> block);   
                virtual void forward();
                virtual ~LayerBlock();
                void summary();
                virtual Type* merge(Type* tensors[]) = 0;
                virtual std::ostream& operator<<(std::ostream& out) {return out;}
                virtual std::istream& operator>>(std::istream& ifs) {return ifs;}      

        };
    } // namespace Layers
} // namespace emptyNN