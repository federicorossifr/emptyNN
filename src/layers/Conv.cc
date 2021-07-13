#include "emptyNN/layers/Conv.hpp"
#include <emptyNN/utils/tensor_utils.hpp>
#include <iostream>
namespace emptyNN {
    namespace Layers {


        template <class Type>
        Conv<Type>::Conv(Shape in, Shape out, ConvParams cp,Activation<Type>* a): Layer<Type>(in,out,a),f_shape(cp.filter),params(cp) {
            filter = Tensor<Type>(cp.filter.height*cp.filter.width*cp.filter.depth*cp.kernels);
        }

        template <class Type>
        Conv<Type>::~Conv() {
        }

        template <class Type>
        Conv<Type>::Conv(Shape in, ConvParams cp,Activation<Type>* a): Layer<Type>(in,a),f_shape(cp.filter),params(cp) {
            size_t filter_size = cp.filter.height*cp.filter.width*cp.filter.depth;
            filter_size*= (cp.isDepthWise)? 1:cp.kernels;
            filter = Tensor<Type>(filter_size);

            std::fill(filter.begin(),filter.end(),0x1);

            emptyNN::Utils::Tensors::fillRandomUniform<Type>(filter,filter_size);

            bias = Tensor<Type>(cp.kernels);
            emptyNN::Utils::Tensors::fillRandomUniform<Type>(bias,cp.kernels);


            size_t des_o_width = (cp.padding == PaddingType::SAME)? in.width : ceil(float(in.width - cp.filter.width+1)/cp.stride);
            size_t des_o_height = (cp.padding == PaddingType::SAME)? in.height : ceil(float(in.height - cp.filter.height+1)/cp.stride);
            
            Shape req_padding = {
                in.width*(cp.stride - 1)+cp.filter.width - 1,
                in.height*(cp.stride - 1)+cp.filter.height - 1
            };

            if (cp.padding == PaddingType::SAME) {
                this->padding = req_padding;

                // Expand the input tensor buffer to the padded dimension
                this->i_tensor.resize(( in.width + req_padding.width )*( in.height + req_padding.height )*in.depth);
            }
            else this->padding = {0,0,0};
            Shape out = {des_o_width,
                         des_o_height,
                         cp.kernels};

            this->o_tensor = Tensor<Type>(out.width*out.height*out.depth);
            this->o_shape = out;
        }        

        template <class Type>
        std::ostream& Conv<Type>::operator<<(std::ostream& out) {
            size_t filter_size = params.filter.height*params.filter.width*params.filter.depth;
            filter_size*= (params.isDepthWise)? 1:params.kernels;
            out.write(reinterpret_cast<char*>(filter),filter_size*sizeof(Type));
            out.write(reinterpret_cast<char*>(bias),params.kernels*sizeof(Type));
            return out;
        }

        template <class Type>
        std::istream& Conv<Type>::operator>>(std::istream& ifs) {
            size_t filter_size = params.filter.height*params.filter.width*params.filter.depth;
            filter_size*= (params.isDepthWise)? 1:params.kernels;            
            ifs.read(reinterpret_cast<char*>(filter),filter_size*sizeof(Type))
               .read(reinterpret_cast<char*>(bias),params.kernels*sizeof(Type));
            return ifs;
        }        
        REGISTER_CLASS(Conv,float)

    }


    
}