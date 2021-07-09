#include <emptyNN/layers/LayerBlock.hpp>

namespace emptyNN {
    namespace Layers {

        template <class Type>
        LayerBlock<Type>::LayerBlock(Shape in, Shape out): Layer<Type>(in,out) {
            
        }

        template <class Type>
        LayerBlock<Type>::LayerBlock(Shape in, Shape out,std::vector<std::vector<Layer<Type>*>> _block): Layer<Type>(in,out) {
            for(auto v: _block) {
                // ToDo: sanity check coherency of input shapes
                // Output shapes may not always need to be the same 
                // This check can be handled by subclasses in the merge
                // method
                block.push_back(v);
            }

        }

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

        template <class Type>
        void LayerBlock<Type>::summary() {
            std::cout << this << "In: (" << this->getInputShape().width << ", " << this->getInputShape().height << "," << this->getInputShape().depth << ")" 
            << " Out: (" << this->getOutputShape().width << ", " << this->getOutputShape().height << "," << this->getOutputShape().depth << ")" <<std::endl;            
            std::cout << "vvv" << std::endl;
           for(size_t i = 0; i < block.size(); ++i) {
                for(Layer<Type>* l: block[i]) {
                    l->summary();
                }
                std::cout << "--\n";
            }                            
        }

        REGISTER_CLASS(LayerBlock,float);


    } // namespace Layers
} // namespace emptyNN