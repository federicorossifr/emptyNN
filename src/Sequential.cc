#include "emptyNN/Sequential.hpp"
#include <cassert>
namespace emptyNN
{
    template <class Type>
    Sequential<Type>::Sequential(std::string&& name): Model(std::move(name)) {}

    template <class Type>
    Sequential<Type>::~Sequential() {
        for(Layer<Type>* l: layers)
            delete(l);
    }

    template <class Type>    
    void Sequential<Type>::stackLayer(Layer<Type>* layer) {
        
        if(layers.size() > 0) {
            if( ! (layers.back()->getOutputShape() == layer->getInputShape()) ) {
                std::cout << "Last Layer: " << layers.back()->getOutputShape().width << " " << layers.back()->getOutputShape().height << " " << layers.back()->getOutputShape().depth << std::endl;
                std::cout << "New Layer: " << layer->getInputShape().width << " " << layer->getInputShape().height << " " << layer->getInputShape().depth << std::endl;            
                assert(layers.back()->getOutputShape() == layer->getInputShape());
            }
        }
        layers.push_back(layer);
    }

    template <class Type>    
    void Sequential<Type>::stackLayers(std::vector<Layer<Type>*> group) {
        for(Layer<Type>* l: group)
            stackLayer(l);
    }    

    template <class Type>    
    Type* Sequential<Type>::predict(Type* in_tensor) {
        Type* handle = in_tensor, 
            *out_tensor = new Type[layers.back()->getOutputShape().size()];
        for(Layer<Type>* l: layers) {
            l->summary();
            l->fillInTensor(handle);
            handle = (*l)();
        }
        return out_tensor;
    }    

    template <class Type>
    Shape Sequential<Type>::getInputShape() {
        return layers.front()->getOutputShape();
    }    

    template <class Type>
    Shape Sequential<Type>::getOutputShape() {
        return layers.back()->getOutputShape();
    }        

    template <class Type>    
    void Sequential<Type>::serialize(std::ofstream& out) {}

    template <class Type>    
    void Sequential<Type>::deserialize(std::ifstream& in) {}

    template <class Type>
    void Sequential<Type>::summary() {
        for(Layer<Type>* l:layers) {
            l->summary();
        }
    }

    template class Sequential<float>;
} // namespace emptyNN
