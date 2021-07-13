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
            void fillRandom(Type* tensor,size_t tensor_size,std::function<Type()> gen) {
                std::generate(tensor,tensor+tensor_size,gen); //d
            }

            template <class Type>
            void fillRandomUniform(Type* tensor,size_t tensor_size,double min=-1.,double max=1.) {
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
