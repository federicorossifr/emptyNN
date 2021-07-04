#pragma once
#include <emptyNN/Model.hpp>
#include <emptyNN/Layer.hpp>
#include <vector>
namespace emptyNN
{
    template <class Type>
    class Sequential: public Model {
        std::vector<Layer<Type>*> layers; 
        public:
            Sequential(std::string&& name);
            ~Sequential();
            virtual void serialize(std::ofstream& out);
            virtual void deserialize(std::ifstream& in);      

            Type* predict(Type* in_tensor);      
            Shape getInputShape();
            Shape getOutputShape();
            void stackLayer(Layer<Type>* layer);
    };
} // namespace emptyNN
