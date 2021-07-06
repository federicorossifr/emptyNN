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

            template Layer<float>* Convolution(Shape in,ConvParams params,Activation<float>* a,Device device);
            template Layer<float>* Dense(Shape in,Shape out,Activation<float>* a,Device device);
            template Layer<float>* BatchNorm(Shape in,float mu,float sigma,Activation<float>* a,Device device);
            template Layer<float>* MaxPool(Shape in,PoolParams params,Activation<float>* a,Device device);
            template Layer<float>* ResBlock(Shape in,Shape out,ResBlockParams params, Device device);


        }

        namespace Activations {

            template <class Type>
            Activation<Type>* Elu(Type alpha) {
                return new emptyNN::Activations::EluFunctor<Type>(alpha);
            };

            template Activation<float>* Elu(float alpha);
        
        }        


    }
}