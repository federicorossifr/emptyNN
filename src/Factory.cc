#include "emptyNN/Factory.hpp"

namespace emptyNN {
    namespace Factory {
        namespace Layers {
            
            template <class Type>
            Layer<Type>* Convolution(Shape in,ConvParams params,Activation<Type>* a,Device device) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::ConvCPUImpl<Type>(in,params,a);
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
            Layer<Type>* Dense(Shape in,Shape out,Activation<Type>* a,Device device) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::DenseCPUImpl<Type>(in,out,a);
                
                    default:
                        throw DeviceNotAllowed(device);
                }

            }            
            
            template <class Type>
            Layer<Type>* BatchNorm(Shape in, Type mu, Type sigma, Activation<Type>* a,Device device) {
                switch (device)
                {
                    case CPU:
                        return new emptyNN::Layers::Impl::BatchNormCPUImpl<Type>(in,mu,sigma,a);
                
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

            REGISTER_FACTORY_LAYER(float)
            
            #ifdef USE_POSIT
            REGISTER_FACTORY_LAYER(Posit16_0)
            REGISTER_FACTORY_LAYER(Posit8_0)
            #endif

        }

        namespace Activations {

            template <class Type>
            Activation<Type>* Elu(Type alpha) {
                return new emptyNN::Activations::EluFunctor<Type>(alpha);
            };

            template Activation<float>* Elu(float alpha);

            #ifdef USE_POSIT
            template Activation<Posit16_0>* Elu(Posit16_0 alpha);
            template Activation<Posit16_1>* Elu(Posit16_1 alpha);
            template Activation<Posit8_0>* Elu(Posit8_0 alpha);
            #endif
        }        


    }
}