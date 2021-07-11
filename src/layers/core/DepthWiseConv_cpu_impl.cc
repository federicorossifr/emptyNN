#include "emptyNN/layers/core/DepthWiseConv_cpu_impl.hpp"
#include <iostream>
#include <cassert>
namespace emptyNN {
    namespace Layers {
        namespace Impl {
            template <class Type>
            DWConvCPUImpl<Type>::DWConvCPUImpl(Shape in, Shape out, ConvParams cp,Activation<Type>* a): Conv<Type>(in,out,cp,a) {
                // Sanity checks, 
                // Number of image channels, filter channels and output channels
                // must all be the same
                assert(in.depth == cp.filter.depth && cp.filter.depth == cp.kernels);
            }

            template <class Type>
            DWConvCPUImpl<Type>::DWConvCPUImpl(Shape in, ConvParams cp,Activation<Type>* a): Conv<Type>(in,cp,a) {
                assert(in.depth == cp.filter.depth && cp.filter.depth == cp.kernels);
                assert(cp.isDepthWise == true);
            }

            template <class Type>
            DWConvCPUImpl<Type>::~DWConvCPUImpl() {}

            template <class Type>
            void DWConvCPUImpl<Type>::forward() {
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
                    // The number of kernel channels is the same as 
                    // Filter and image channels, so we pin the depth for 
                    // all of them here and we do not iterate over the depth later
                    // Each filter channel is matched to only one image channel
                    // and to only one output channel
                    Type* o_pin = &o_tensor[kernel*(out.height*out.width)];
                    Type* i_pin = &i_tensor[kernel*(in.height*in.width)];
                    Type* f_pin = &filter[kernel*(fil.height*fil.width)];

                    for(size_t i = 0; i < out.height; ++i) {
                        for(size_t j = 0; j < out.width; ++j) {
                            
                            Type accum(0);

                            for(size_t k = 0; k < fil.height; ++k) {
                                for(size_t l = 0; l < fil.width; ++l) {
                                    Type _a = f_pin[ k*fil.width + l ];
                                    Type _b = i_pin[ (i*stride+k)*in.width + l+j*stride ];
                                    accum +=  _a * _b;
                                }
                            }

                            o_pin[i*out.width+j] = accum;

                        }
                    }
                }
            }

     

            template <class Type>
            void DWConvCPUImpl<Type>::backward() {}            
                

            REGISTER_CLASS(DWConvCPUImpl,float)

        }
    }



}