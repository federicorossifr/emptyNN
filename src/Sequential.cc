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
        if(layers.size() > 0) 
            assert(layers.back()->getOutputShape() == layer->getInputShape());
        layers.push_back(layer);
    }

    template <class Type>    
    Type* Sequential<Type>::predict(Type* in_tensor) {
        Type* handle = in_tensor, 
            *out_tensor = new Type[layers.back()->getOutputShape().size()];
        for(Layer<Type>* l: layers) {
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

    

    template class Sequential<float>;
} // namespace emptyNN
