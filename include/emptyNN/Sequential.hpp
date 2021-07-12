#pragma once
#include <emptyNN/Model.hpp>
#include <emptyNN/Layer.hpp>
#include <vector>
namespace emptyNN
{
    namespace io
    {
        template <class Type>
        class BinarySerializer;        
    } // namespace io
    

    
    template <class Type>
    class Sequential: public Model {
        std::vector<Layer<Type>*> layers; 
        public:
            Sequential(std::string&& name);
            ~Sequential();
            virtual void serialize(std::ofstream& out);
            virtual void deserialize(std::ifstream& in);      
            void summary();

            Type* predict(Type* in_tensor);      
            Shape getInputShape();
            Shape getOutputShape();
            void stackLayer(Layer<Type>* layer);
            void stackLayers(std::vector<Layer<Type>*> group);

            friend class io::BinarySerializer<Type>;
    };
} // namespace emptyNN
