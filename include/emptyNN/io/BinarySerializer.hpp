#pragma once
#include <emptyNN/io/Serializer.hpp>
#include <string>
#include <fstream>
namespace emptyNN
{


    namespace io
    {

        template <class Type>        
        class BinarySerializer: public Serializer<Type> {
            public:
            BinarySerializer(std::string filename);
            
            virtual void serialize(Sequential<Type>* model);

            virtual Sequential<Type>* deserialize();            

        };        
    } // namespace io
    

 
} // namespace EmptyNN

