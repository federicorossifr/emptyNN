#include "emptyNN/layers/core/BatchNorm_cpu_impl.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            BatchNormCPUImpl<Type>::BatchNormCPUImpl(Shape in, Type mu, Type sigma, Activation<Type>* a): BatchNormalization<Type>(in,mu,sigma,a){}

            template <class Type>
            BatchNormCPUImpl<Type>::~BatchNormCPUImpl() {}

            template <class Type>
            void BatchNormCPUImpl<Type>::forward() {
                Type mu = this->mu;
                Type sigma = this->sigma;
                Type* i_tensor = this->i_tensor;
                Type* o_tensor = this->o_tensor;
                size_t in_size = this->i_shape.size();

                #pragma omp parallel for simd
                for(int i = 0; i < in_size; ++i)
                    o_tensor[i] = (i_tensor[i]-mu)/sigma;
            }

            template <class Type>
            void BatchNormCPUImpl<Type>::activate() {
                Activation<Type>* a = this->activation;
                Shape out = this->o_shape;
                Type* o_tensor = this->o_tensor;
                if( a == nullptr) return;

                #pragma omp parallel for simd
                for(size_t i = 0; i < out.size(); ++i) {
                    o_tensor[i] = (*a)(o_tensor[i]);
                }
            }

            template <class Type>
            void BatchNormCPUImpl<Type>::backward() {}            

            REGISTER_CLASS(BatchNormCPUImpl,float);
       
        }

    }



}