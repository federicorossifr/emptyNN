#include "emptyNN/layers/Conv.hpp"
#include <iostream>
namespace emptyNN {
    namespace Layers {


        template <class Type>
        Conv<Type>::Conv(Shape in, Shape out, ConvParams cp,Activation<Type>* a): Layer<Type>(in,out,a),f_shape(cp.filter),params(cp) {
            filter = new Type[cp.filter.height*cp.filter.width*cp.filter.depth*cp.kernels];
        }

        template <class Type>
        Conv<Type>::~Conv() {
            delete[] filter;
        }

        template <class Type>
        Conv<Type>::Conv(Shape in, ConvParams cp,Activation<Type>* a): Layer<Type>(in,a),f_shape(cp.filter),params(cp) {
            size_t filter_size = cp.filter.height*cp.filter.width*cp.filter.depth;
            filter_size*= (cp.isDepthWise)? 1:cp.kernels;
            filter = new Type[filter_size];

            std::fill(filter,filter+filter_size,0x1);

            size_t des_o_width = (cp.padding == PaddingType::SAME)? in.width : ceil(float(in.width - cp.filter.width+1)/cp.stride);
            size_t des_o_height = (cp.padding == PaddingType::SAME)? in.height : ceil(float(in.height - cp.filter.height+1)/cp.stride);
            
            Shape req_padding = {
                in.width*(cp.stride - 1)+cp.filter.width - 1,
                in.height*(cp.stride - 1)+cp.filter.height - 1
            };

            if (cp.padding == PaddingType::SAME) {
                this->padding = req_padding;
                delete[] this->i_tensor;

                // Expand the input tensor buffer to the padded dimension
                this->i_tensor = new Type[ ( in.width + req_padding.width )*( in.height + req_padding.height )*in.depth ];
            }
            else this->padding = {0,0,0};
            Shape out = {des_o_width,
                         des_o_height,
                         cp.kernels};

            this->o_tensor = new Type[out.width*out.height*out.depth];
            this->o_shape = out;
        }        
        REGISTER_CLASS(Conv,float)

    }


    
}