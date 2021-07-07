#include "emptyNN/layers/core/MaxPool_cpu_impl.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            MaxPoolCPUImpl<Type>::MaxPoolCPUImpl(Shape in,PoolParams params,Activation<Type>* a): MaxPooling<Type>(in,params,a){}

            template <class Type>
            MaxPoolCPUImpl<Type>::~MaxPoolCPUImpl() {}

            template <class Type>
            void MaxPoolCPUImpl<Type>::forward() {
                Shape out = this->o_shape;
                Shape in = this->i_shape;
                PoolParams params = this->params;
                Type* i_tensor = this->i_tensor;
                Type* o_tensor = this->o_tensor;
                
                #pragma omp parallel for
                for(size_t depth = 0; depth < in.depth; ++depth) {
                    Type* i_pin = &i_tensor[depth*(in.height*in.width)];
                    Type* o_pin = &i_tensor[depth*(out.height*out.width)];

                    for(size_t i = 0; i < out.height; ++i) {
                        for(size_t j = 0; j < out.width; ++j) {

                            Type max_ = i_pin[0];

                            for(size_t l = 0; l < params.factor.height; ++l ) {
                                for(size_t k = 0; k < params.factor.width; ++k) {
                                    Type pun = i_pin[ (i*params.stride+k)*in.width + l+j*params.stride ];
                                    max_ = std::max(max_, pun);
                                }
                            }

                            o_pin[i*out.width + j] = max_;

                        }
                    }


                }
                
            }

            template <class Type>
            void MaxPoolCPUImpl<Type>::activate() {
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
            void MaxPoolCPUImpl<Type>::backward() {}            

            REGISTER_CLASS(MaxPoolCPUImpl,float);
       
        }
    }



}