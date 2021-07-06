#include <emptyNN/layers/LayerBlock.hpp>

namespace emptyNN {
    namespace Layers {

        template <class Type>
        LayerBlock<Type>::LayerBlock(Shape in, Shape out): Layer<Type>(in,out) {}

        template <class Type>
        void LayerBlock<Type>::forward() {
            Type* i_tensor = this->i_tensor;
            Type* o_tensor = this->o_tensor;
            Type** gathered_tensors = new Type*[block.size()];

            for(size_t i = 0; i < block.size(); ++i) {
                Type* handle = i_tensor;
                for(Layer<Type>* l: block[i]) {
                    l->fillInTensor(handle);
                    handle = (*l)();
                }
                gathered_tensors[i] = handle;
            }
            
            o_tensor = merge(gathered_tensors);
            delete[] gathered_tensors;
        }

        template <class Type>
        LayerBlock<Type>::~LayerBlock() {
            for(size_t i = 0; i < block.size(); ++i) {
                for(Layer<Type>* l: block[i]) {
                    delete(l);
                }
            }
        }

        template class LayerBlock<float>;


    } // namespace Layers
} // namespace emptyNN