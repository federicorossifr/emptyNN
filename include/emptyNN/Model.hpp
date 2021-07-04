#pragma once
#include <string>
#include <iostream>
namespace emptyNN {
     
     class Model {
        std::string name;

        public:
            Model(std::string&& name);
            virtual void serialize(std::ofstream& out) = 0;
            virtual void deserialize(std::ifstream& in) = 0;


     };


}