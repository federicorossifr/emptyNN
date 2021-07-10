#include "emptyNN/layers/core/Dense_cpu_impl.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            DenseCPUImpl<Type>::DenseCPUImpl(Shape in, Shape out,Activation<Type>* a): Dense<Type>(in,out,a){}

            template <class Type>
            DenseCPUImpl<Type>::~DenseCPUImpl() {}

            template <class Type>
            void DenseCPUImpl<Type>::forward() {
                Shape out = this->o_shape;
                Shape in = this->i_shape;
                Type* i_tensor = this->i_tensor;
                Type* o_tensor = this->o_tensor;
                Type* fc       = this->connections;
                size_t fc_rows = in.size(),
                       fc_cols = out.size();
                for(size_t i = 0; i < fc_cols; ++i) {
                    Type accum = 0.;

                    for(size_t j = 0; j < fc_rows; ++j) {
                        
                        accum += i_tensor[j]*fc[j*fc_cols+i];
                    }

                    o_tensor[i] = accum;
                }

            }

            template <class Type>
            void DenseCPUImpl<Type>::activate() {
                Activation<Type>* a = this->activation;
                Shape out = this->o_shape;
                Type* o_tensor = this->o_tensor;
                if( a == nullptr) return;

                (*a)(o_tensor,out);
            }

            template <class Type>
            void DenseCPUImpl<Type>::backward() {}            

            REGISTER_CLASS(DenseCPUImpl,float);
       
        }
    }



}