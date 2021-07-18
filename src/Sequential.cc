/*
emptyNN
Copyright (C) 2021 Federico Rossi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
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
            l->fillInTensor(handle);
            handle = (*l)();
        }
        return out_tensor;
    }

    template <class Type>
    void Sequential<Type>::fit(Type* in_tensor) {
        Type* prediction = this->predict(in_tensor);
        Type* gradHandle;
        for(auto it = layers.end(); it != layers.begin(); --it) {
            (*it)->backward(prediction);
        }
    }

    template <class Type>
    Shape Sequential<Type>::getInputShape() {
        return layers.front()->getInputShape();
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

    REGISTER_CLASS(Sequential,float);
} // namespace emptyNN
