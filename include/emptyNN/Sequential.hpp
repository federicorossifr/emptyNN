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
#pragma once
#include <emptyNN/Model.hpp>
#include <emptyNN/Layer.hpp>
#include <vector>
namespace emptyNN
{
    namespace io
    {
        template <class Type>
        class Serializer;        
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
            void  fit(Type* in_tensor/* ToDo Label */);
            Shape getInputShape();
            Shape getOutputShape();
            void stackLayer(Layer<Type>* layer);
            void stackLayers(std::vector<Layer<Type>*> group);

            friend class io::Serializer<Type>;
    };
} // namespace emptyNN
