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
            template Layer<float>* Convolution(Shape in,ConvParams params,Activation<float>* a,Device device);


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