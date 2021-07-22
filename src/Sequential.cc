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
    Tensor<Type> Sequential<Type>::predict(Tensor<Type> in_tensor) {
        Tensor<Type>& handle = in_tensor,
            out_tensor = Tensor<Type>(layers.back()->getOutputShape().size());
        for(Layer<Type>* l: layers) {
            l->fillInTensor(handle);
            handle = (*l)();
        }
        return out_tensor;
    }

    template <class Type>
    void Sequential<Type>::fit(Tensor<Type> in_tensor, Tensor<Type> truth) {
        Tensor<Type> prediction = this->predict(in_tensor);
        size_t out_size = layers.back()->getOutputShape().size();
        std::transform(truth.begin(),truth.end(),prediction.begin(),prediction.begin(),std::minus<float>());
        Type loss = std::accumulate(truth.begin(),truth.end(),Type(0.),[](Type a, Type b) {
            return a + (Type)std::pow((double)b,2);
        });

        Tensor<Type> gradHandle = prediction;
        std::cout << "Loss: " << std::sqrt(loss) << std::endl;
        for(auto it = layers.rbegin(); it != layers.rend(); ++it) {
            gradHandle = (*it)->backward(gradHandle);
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
