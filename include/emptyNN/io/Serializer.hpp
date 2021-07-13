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
            
            virtual void dumpBinaryWeights(Sequential<Type>* model);

            virtual void loadBinaryWeights(Sequential<Type>* model);            




        };        
    } // namespace io
    

 
} // namespace EmptyNN

