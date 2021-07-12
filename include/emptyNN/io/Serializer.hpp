#pragma once
#include <emptyNN/Sequential.hpp>
#include <string>
#include <fstream>
namespace emptyNN
{
    template <class Type>
    class Sequential;
    
    namespace io
    {

        template <class Type>        
        class Serializer {
            protected:
            std::string filename;


            public:
            Serializer(std::string filename);
            
            virtual void serialize(Sequential<Type>* model) = 0;

            virtual Sequential<Type>* deserialize() = 0;            




        };        
    } // namespace io
    

 
} // namespace EmptyNN

