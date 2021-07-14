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
#include "emptyNN/layers/core/Conv_cpu_impl.hpp"
#include "emptyNN/layers/core/DepthWiseConv_cpu_impl.hpp"
#include "emptyNN/layers/core/Conv_rvv_impl.hpp"
#include "emptyNN/layers/core/Dense_cpu_impl.hpp"
#include "emptyNN/layers/core/BatchNorm_cpu_impl.hpp"
#include "emptyNN/layers/core/MaxPool_cpu_impl.hpp"
#include "emptyNN/layers/core/ResBlock_cpu_impl.hpp"
#include "emptyNN/layers/core/Concat_cpu_impl.hpp"
#include "emptyNN/layers/core/Add_cpu_impl.hpp"
#include "emptyNN/layers/Flatten.hpp"
#include "emptyNN/layers/Pad.hpp"
#include "emptyNN/activations/Elu.hpp"

namespace emptyNN {
    namespace Factory {
        namespace Layers {
            
            template <class Type>
            Layer<Type>* Convolution(Shape in,ConvParams params,Activation<Type>* a,Device device);

            template <class Type>
            Layer<Type>* DWConvolution(Shape in,ConvParams params,Activation<Type>* a,Device device);            

            template <class Type>
            Layer<Type>* Dense(Shape in,Shape out,Activation<Type>* a,Device device);

            template <class Type>
            Layer<Type>* BatchNorm(Shape in, double mu, double sigma, Activation<Type>* a, Device device);

            template <class Type>
            Layer<Type>* MaxPool(Shape in, PoolParams params, Activation<Type>* a, Device device);

            template <class Type>
            Layer<Type>* ResBlock(Shape in,Shape out,ResBlockParams params, Device device);            

            template <class Type>
            Layer<Type>* Concat(Shape in, Shape out,std::vector<std::vector<Layer<Type>*>> _block,Device device);          

            template <class Type>
            Layer<Type>* Add(Shape in, Shape out,std::vector<std::vector<Layer<Type>*>> _block,Device device);                   

            template <class Type>
            Layer<Type>* Flatten(Shape in, Device device);                          

            template <class Type>
            Layer<Type>* Pad(Shape in, Shape out, Device device);       

        }

        namespace Activations {
            template <class Type>
            Activation<Type>* Elu(double alpha);
        }

        #define REGISTER_FACTORY_LAYER(TYPE) \
            template Layer<TYPE>* Convolution(Shape in,ConvParams params,Activation<TYPE>* a,Device device); \
            template Layer<TYPE>* DWConvolution(Shape in,ConvParams params,Activation<TYPE>* a,Device device); \
            template Layer<TYPE>* Dense(Shape in,Shape out,Activation<TYPE>* a,Device device); \
            template Layer<TYPE>* BatchNorm(Shape in,double mu,double sigma,Activation<TYPE>* a,Device device); \
            template Layer<TYPE>* MaxPool(Shape in,PoolParams params,Activation<TYPE>* a,Device device);  \
            template Layer<TYPE>* ResBlock(Shape in,Shape out,ResBlockParams params, Device device); \
            template Layer<TYPE>* Concat(Shape in, Shape out,std::vector<std::vector<Layer<TYPE>*>> _block,Device device); \
            template Layer<TYPE>* Add(Shape in, Shape out,std::vector<std::vector<Layer<TYPE>*>> _block,Device device); \
            template Layer<TYPE>* Flatten(Shape in, Device device); \
            template Layer<TYPE>* Pad(Shape in, Shape out, Device device); 
    }
}