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
#include "emptyNN/Factory.hpp"

namespace emptyNN {
    namespace Factory {
        namespace Layers {
            
            template <class Type>
            Layer<Type>* Convolution(Shape in,ConvParams params,Activation<Type>* a,Device device, bool withBias) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::ConvCPUImpl<Type>(in,params,a,withBias);
                    case CPU_RVV:
                        #ifdef USE_RVV
                            return new emptyNN::Layers::Impl::ConvRVVImpl<Type>(in,params,a);
                        #else 
                            throw DeviceNotAllowed(device);
                        #endif

                    default:
                        throw DeviceNotAllowed(device);
                }

            }     

            template <class Type>
            Layer<Type>* DWConvolution(Shape in,ConvParams params,Activation<Type>* a,Device device) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::DWConvCPUImpl<Type>(in,params,a);
                    case CPU_RVV:
                        #ifdef USE_RVV

                        #else 
                            throw DeviceNotAllowed(device);
                        #endif

                    default:
                        throw DeviceNotAllowed(device);
                }

            }                        

            template <class Type>
            Layer<Type>* Dense(Shape in,Shape out,Activation<Type>* a,Device device,bool withBias) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::DenseCPUImpl<Type>(in,out,a,withBias);
                
                    default:
                        throw DeviceNotAllowed(device);
                }

            }            
            
            template <class Type>
            Layer<Type>* BatchNorm(Shape in, double mu, double sigma, Activation<Type>* a,Device device) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::BatchNormCPUImpl<Type>(in,Type(mu),Type(sigma),a);
                
                    default:
                        throw DeviceNotAllowed(device);
                }

            }  

            template <class Type>
            Layer<Type>* MaxPool(Shape in, PoolParams params, Activation<Type>* a,Device device) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::MaxPoolCPUImpl<Type>(in,params,a);
                
                    default:
                        throw DeviceNotAllowed(device);
                }

            }                    

            template <class Type>
            Layer<Type>* ResBlock(Shape in,Shape out,ResBlockParams params, Device device) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::ResidualBlockCPUImpl<Type>(in,out,params);
                
                    default:
                        throw DeviceNotAllowed(device);
                }

            }   

            template <class Type>
            Layer<Type>* Concat(Shape in, Shape out,std::vector<std::vector<Layer<Type>*>> _block, Device device) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::ConcatCPUImpl<Type>(in,out,_block);
                
                    default:
                        throw DeviceNotAllowed(device);
                }                
            }

            template <class Type>
            Layer<Type>* Add(Shape in, Shape out,std::vector<std::vector<Layer<Type>*>> _block, Device device) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::CombineMergeCPUImpl<Type>(in,out,_block,Type(0.),[](Type& a,Type& b){
                            return a+b;
                        });
                
                    default:
                        throw DeviceNotAllowed(device);
                }                
            }  

            template <class Type>
            Layer<Type>* Multiply(Shape in, Shape out,std::vector<std::vector<Layer<Type>*>> _block, Device device) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::CombineMergeCPUImpl<Type>(in,out,_block,Type(1.),[](Type& a,Type& b){
                            return a*b;
                        });
                
                    default:
                        throw DeviceNotAllowed(device);
                }                
            }                        

            template <class Type>
            Layer<Type>* Flatten(Shape in, Device device) {
                switch (device)
                {
                    case CPU:
                    case GPU:
                    case CPU_RVV:
                    case CPU_SVE:
                        return new emptyNN::Layers::Flatten<Type>(in);
                
                    default:
                        throw DeviceNotAllowed(device);
                }                      
            }

            template <class Type>
            Layer<Type>* Pad(Shape in, Shape out, Device device) {
                switch (device)
                {
                    case CPU:
                    case GPU:
                    case CPU_RVV:
                    case CPU_SVE:
                        return new emptyNN::Layers::Pad<Type>(in,out);
                
                    default:
                        throw DeviceNotAllowed(device);
                }                      
            }                            

            REGISTER_FACTORY_LAYER(float)
            
            #ifdef USE_POSIT
            REGISTER_FACTORY_LAYER(Posit16_1)
            REGISTER_FACTORY_LAYER(Posit16_0)
            REGISTER_FACTORY_LAYER(Posit8_0)
            REGISTER_FACTORY_LAYER(Bfloat16)
            REGISTER_FACTORY_LAYER(Bfloat8)
            REGISTER_FACTORY_LAYER(FloatEmu)
            #endif

        }

        namespace Activations {

            template <class Type>
            Activation<Type>* Elu(double alpha) {
                return new emptyNN::Activations::EluFunctor<Type>(Type(alpha));
            };

            template Activation<float>* Elu(double alpha);

            #ifdef USE_POSIT
            template Activation<Posit16_0>* Elu( double alpha);
            template Activation<Posit16_1>* Elu( double alpha);
            template Activation<Posit8_0>* Elu( double alpha);
            template Activation<Bfloat16>* Elu( double alpha);
            template Activation<Bfloat8>* Elu( double alpha);
            template Activation<FloatEmu>* Elu( double alpha);
            #endif
        }        


    }
}