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
        class Conv: public Layer<Type> {
            protected:
                Tensor<Type> filter;
                Tensor<Type> bias;
                Shape f_shape;
                ConvParams params;
                Shape padding;
                const bool hasBias;
            public:
                virtual ~Conv();
                Conv(Shape in, ConvParams cp,Activation<Type>* a, bool withBias = true);
                virtual void summary() {
                    size_t filter_size = params.filter.height*params.filter.width*params.filter.depth;
                    filter_size*= (params.isDepthWise)? 1:params.kernels;                    
                    Layer<Type>::summary(); std::cout << "Conv weights: " << filter_size + params.kernels << std::endl; 
                }
                virtual std::ostream& operator<<(std::ostream& out);
                virtual std::istream& operator>>(std::istream& ifs);                


        };
    }
}