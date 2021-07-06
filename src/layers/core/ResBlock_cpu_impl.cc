#include <emptyNN/layers/core/ResBlock_cpu_impl.hpp>
#include <emptyNN/layers/core/Conv_cpu_impl.hpp>
#include <emptyNN/activations/Elu.hpp>

namespace emptyNN {
    namespace Layers {
        namespace Impl {

            template <class Type>
            ResidualBlockCPUImpl<Type>::ResidualBlockCPUImpl(Shape in,Shape out,ResBlockParams params): LayerBlock<Type>(in,out),params(params) {
                std::vector<Layer<Type>*> convLayers;
                using ELU = emptyNN::Activations::EluFunctor<Type>;
                Shape middle;
                if(params.halve)
                    convLayers.push_back(new ConvCPUImpl<Type>( in, { {3,3,in.depth}, 2*in.depth, 2, PaddingType::ZERO }  , new ELU(1.)) );
                else 
                    convLayers.push_back(new ConvCPUImpl<Type>( in, { {3,3,in.depth}, in.depth, 1, PaddingType::SAME }  , new ELU(1.)) );
                
                middle = convLayers.back()->getOutputShape();

                for(size_t i = 1; i < params.block_size; ++i) {
                    convLayers.push_back(new ConvCPUImpl<Type>( middle, { {3,3,middle.depth}, middle.depth, 1, PaddingType::SAME }  , new ELU(1.)) );
                }

                this->block.push_back(convLayers);
            }

            template <class Type>
            Type* ResidualBlockCPUImpl<Type>::merge(Type* tensors[]) {
                // ResNets have only one block filled with layers, the other branch is just a shortcut
                // so we take the tansor from in_tensor
                Type* shortcut = this->i_tensor;
                Type* conv_tensor = tensors[0];
                Type* o_tensor = this->o_tensor;
                Shape in = this->i_shape;
                Shape out = this->o_shape;

                #pragma omp parallel for
                for(size_t i = 0; i < in.depth; ++i ) { // In-depth is <= out depth
                    Type* i_pin = &shortcut[i*(in.width*in.height)];
                    Type* o_pin = &o_tensor[i*(out.width*out.height)];

                    size_t stride = (params.halve && params.identity) ? 2:1;
                    size_t linearized_size = out.width*out.height;
                    for(size_t j = 0; j < linearized_size; ++j) {
                        Type _a = conv_tensor[j];
                        Type _b = shortcut[stride*j];
                        o_pin[j] = _a + _b;
                    }

                }
                
                return o_tensor;
            }   

            REGISTER_CLASS(ResidualBlockCPUImpl,float);

        } // namespace Impl
    } // namespace Layers
} // namespace emptyNN
