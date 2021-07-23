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
#include <emptyNN/io/Serializer.hpp>

namespace emptyNN
{
    namespace io
    {
        template <class Type>
        Serializer<Type>::Serializer(std::string filename): filename(filename) {}

        

        template <class Type>
        void Serializer<Type>::dumpBinaryWeights(Sequential<Type>* model) {
            std::ofstream of(this->filename, std::ios::binary | std::ios::out);
            for(auto& l: model->layers) {
               *l << of;
            }

        }

        template <class Type>
        void Serializer<Type>::loadBinaryWeights(Sequential<Type>* model) {
            std::ifstream ifs(this->filename, std::ios::binary | std::ios::in);
            for(auto& l: model->layers) {
               *l >> ifs;
            }
        }        
        REGISTER_CLASS(Serializer,float)
    } // namespace io
    
} // namespace emptyNN
