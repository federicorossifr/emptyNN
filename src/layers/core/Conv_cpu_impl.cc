#include "emptyNN/layers/core/Conv_cpu_impl.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            ConvCPUImpl<Type>::ConvCPUImpl(Shape in, Shape out, ConvParams cp,Activation<Type>* a): Conv<Type>(in,out,cp,a){}

            template <class Type>
            ConvCPUImpl<Type>::ConvCPUImpl(Shape in, ConvParams cp,Activation<Type>* a): Conv<Type>(in,cp,a){}

            template <class Type>
            ConvCPUImpl<Type>::~ConvCPUImpl() {}

            template <class Type>
            void ConvCPUImpl<Type>::forward() {
                Shape out = this->o_shape;
                Shape in = this->i_shape;
                Shape fil = this->f_shape;
                Shape padding = this->padding;
                Type* filter = this->filter;
                Type* i_tensor = this->i_tensor;
                Type* o_tensor = this->o_tensor;
                size_t stride = this->params.stride;
                size_t kernels = this->params.kernels;
                #pragma omp parallel for                
                for(size_t kernel = 0; kernel < kernels; ++kernel) {
                    Type* o_pin = &o_tensor[kernel*(out.height*out.width)];
                    Type* i_padded = new Type[ (in.height+padding.height )*(in.width+padding.width )];     
                    std::fill(i_padded,i_padded+( in.height + padding.height )*( in.width + padding.width ), 0x0);                                   
                    for(size_t i = 0; i < out.height; ++i) {
                        for(size_t j = 0; j < out.width; ++j) {
                            Type daccum(0);


                            for(size_t depth = 0; depth < fil.depth; ++depth) {

                                Type* i_pin = &i_tensor[depth*(in.height*in.width)];
                                
                                // ToDo: handle padding in a better way
                                
                                std::copy(i_pin,i_pin+in.height*in.width,i_padded);

                                Type* f_pin = &filter[depth*(fil.height*fil.width)];
                                Type accum(0);

                                for(size_t k = 0; k < fil.height; ++k) {
                                    for(size_t l = 0; l < fil.width; ++l) {
                                        Type _a = f_pin[ k*fil.width + l ];
                                        Type _b = i_padded[ (i*stride+k)*in.width + l+j*stride ];
                                        accum +=  _a * _b;
                                    }
                                }
                                daccum+=accum;
                                
                            }
                            o_pin[i*out.width+j] = daccum;

                        }
                    }
                    delete [] i_padded;                    
                }
            }

            template <class Type>
            void ConvCPUImpl<Type>::activate() {
                Activation<Type>* a = this->activation;
                Shape out = this->o_shape;
                Type* o_tensor = this->o_tensor;
                if( a == nullptr) return;
                for(size_t i = 0; i < out.size(); ++i) {
                    o_tensor[i] = (*a)(o_tensor[i]);
                }
            }

            template <class Type>
            void ConvCPUImpl<Type>::backward() {}            
                
        }
    }



    REGISTER_CONV_CPU_IMPL(float)
}