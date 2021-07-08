#include <emptyNN/layers/core/Concat_cpu_impl.hpp>
#include <emptyNN/layers/core/Conv_cpu_impl.hpp>
#include <emptyNN/activations/Elu.hpp>

namespace emptyNN {
    namespace Layers {
        namespace Impl {

            template <class Type>
            ConcatCPUImpl<Type>::ConcatCPUImpl(Shape in,Shape out,std::vector<std::vector<Layer<Type>*>> _block): LayerBlock<Type>(in,out,_block) {
                // ToDo: sanity checks on output 
                // Output should be sum of all the getOutputShape().size()
                // from layers
                // Maybe make a constructor withot "out" parameter
                // and infer directly the output
            }

            template <class Type>
            Type* ConcatCPUImpl<Type>::merge(Type* tensors[]) {
                // ResNets have only one block filled with layers, the other branch is just a shortcut
                // so we take the tansor from in_tensor
                Type* o_tensor = this->o_tensor;
                Shape in = this->i_shape;
                Shape out = this->o_shape;
                auto blocks = this->block;
                size_t n_tensors = this->block.size();
                #pragma omp parallel for
                for(size_t i = 0; i < n_tensors; ++i ) { // In-depth is <= out depth
                    Shape out_shape_i = blocks[i].back()->getOutputShape();
                    size_t area_i = out_shape_i.size();
                    Type* tensor_i = tensors[i];
                    std::copy(tensor_i,tensor_i+area_i,o_tensor);
                    o_tensor+=area_i;
                }
                
                return o_tensor;
            }   

            REGISTER_CLASS(ConcatCPUImpl,float);

        } // namespace Impl
    } // namespace Layers
} // namespace emptyNN
