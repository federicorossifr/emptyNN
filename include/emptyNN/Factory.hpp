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
#include "emptyNN/layers/core/Combine_cpu_impl.hpp"
#include "emptyNN/layers/Flatten.hpp"
#include "emptyNN/layers/Pad.hpp"
#include "emptyNN/activations/Elu.hpp"
#include <memory>
namespace emptyNN {
    namespace Factory {
        namespace Layers {
            
            template <class Type>
            std::unique_ptr<Layer<Type>> Convolution(Shape in,ConvParams params,std::unique_ptr<Activation<Type>> a,Device device, bool withBias = true);

            template <class Type>
            std::unique_ptr<Layer<Type>> DWConvolution(Shape in,ConvParams params,std::unique_ptr<Activation<Type>> a,Device device);

            template <class Type>
            std::unique_ptr<Layer<Type>> Dense(Shape in,Shape out,std::unique_ptr<Activation<Type>> a,Device device, bool withBias = true);

            template <class Type>
            std::unique_ptr<Layer<Type>> BatchNorm(Shape in, double mu, double sigma, std::unique_ptr<Activation<Type>> a, Device device);

            template <class Type>
            std::unique_ptr<Layer<Type>> MaxPool(Shape in, PoolParams params, std::unique_ptr<Activation<Type>> a, Device device);

            template <class Type>
            std::unique_ptr<Layer<Type>> ResBlock(Shape in,Shape out,ResBlockParams params, Device device);

            template <class Type>
            std::unique_ptr<Layer<Type>> Concat(Shape in, Shape out,std::vector<std::vector<std::unique_ptr<Layer<Type>>>>&& _block,Device device);

            template <class Type>
            std::unique_ptr<Layer<Type>> Add(Shape in, Shape out,std::vector<std::vector<std::unique_ptr<Layer<Type>>>>&& _block,Device device);

            template <class Type>
            std::unique_ptr<Layer<Type>> Multiply(Shape in, Shape out,std::vector<std::vector<std::unique_ptr<Layer<Type>>>>&& _block, Device device);

            template <class Type>
            std::unique_ptr<Layer<Type>> Flatten(Shape in, Device device);

            template <class Type>
            std::unique_ptr<Layer<Type>> Pad(Shape in, Shape out, Device device);

        }

        namespace Activations {
            template <class Type>
            std::unique_ptr<Activation<Type>> Elu(double alpha);
        }

        #define REGISTER_FACTORY_LAYER(TYPE) \
            template std::unique_ptr<Layer<TYPE>> Convolution(Shape in,ConvParams params,std::unique_ptr<Activation<TYPE>> a,Device device,bool withBias); \
            template std::unique_ptr<Layer<TYPE>> DWConvolution(Shape in,ConvParams params,std::unique_ptr<Activation<TYPE>> a,Device device); \
            template std::unique_ptr<Layer<TYPE>> Dense(Shape in,Shape out,std::unique_ptr<Activation<TYPE>> a,Device device,bool withBias); \
            template std::unique_ptr<Layer<TYPE>> BatchNorm(Shape in,double mu,double sigma,std::unique_ptr<Activation<TYPE>> a,Device device); \
            template std::unique_ptr<Layer<TYPE>> MaxPool(Shape in,PoolParams params,std::unique_ptr<Activation<TYPE>> a,Device device);  \
            template std::unique_ptr<Layer<TYPE>> ResBlock(Shape in,Shape out,ResBlockParams params, Device device); \
            template std::unique_ptr<Layer<TYPE>> Concat(Shape in, Shape out,std::vector<std::vector<std::unique_ptr<Layer<TYPE>>>>&& _block,Device device); \
            template std::unique_ptr<Layer<TYPE>> Add(Shape in, Shape out,std::vector<std::vector<std::unique_ptr<Layer<TYPE>>>>&& _block,Device device); \
            template std::unique_ptr<Layer<TYPE>> Multiply(Shape in, Shape out,std::vector<std::vector<std::unique_ptr<Layer<TYPE>>>>&& _block,Device device); \
            template std::unique_ptr<Layer<TYPE>> Flatten(Shape in, Device device); \
            template std::unique_ptr<Layer<TYPE>> Pad(Shape in, Shape out, Device device);
    }
}