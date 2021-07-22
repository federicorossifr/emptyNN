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
#include <random>
#include <iostream>
#include <emptyNN/Types.hpp>
#include <functional>
namespace emptyNN
{
    namespace Utils
    {
        namespace Tensors
        {

            template <class Type>
            void fillRandom(Tensor<Type> tensor,size_t tensor_size,std::function<Type()> gen) {
                std::generate(tensor.begin(),tensor.end(),gen); //d
            }

            template <class Type>
            void fillRandomUniform(Tensor<Type> tensor,size_t tensor_size,double min=-1.,double max=1.) {
                std::mt19937 mtwister;
                std::uniform_real_distribution<> dis(min,max);
                mtwister.seed(Random::globalSeed);                   
                fillRandom<Type>(tensor,tensor_size,[&dis,&mtwister,&min,&max]() -> Type {
                    return Type(dis(mtwister));
                });
            }            



        } // namespace Tensors   
    } // namespace Utils
} // namespace emptyNN
