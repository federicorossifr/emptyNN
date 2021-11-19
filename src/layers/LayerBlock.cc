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
#include <emptyNN/layers/LayerBlock.hpp>

namespace emptyNN {
    namespace Layers {

        template <class Type>
        LayerBlock<Type>::LayerBlock(Shape in, Shape out): Layer<Type>(in,out) {
            
        }


        template <class Type>
        LayerBlock<Type>::LayerBlock(Shape in, Shape out,std::vector<std::vector<std::unique_ptr<Layer<Type>>>>&& _block): Layer<Type>(in,out) {
            auto check_stack = [&in](std::vector<std::unique_ptr<Layer<Type>>>& stack) -> bool {
                Shape a = in;
                for(auto& v: stack) {
                    if(!(v->getInputShape() == a)) {
                        std::cout << "Last Layer: " << a.width << " " << a.height << " " << a.depth << std::endl;
                        std::cout << "New Layer: " << v->getInputShape().width << " " << v->getInputShape().height << " " << v->getInputShape().depth << std::endl;            
                        return false;
                    } 
                    a = v->getOutputShape();
                }
                return true;
            };

            for(auto& v: _block) {
                assert(check_stack(v));
                block.push_back(std::move(v));
            }
        }

        template <class Type>
        void LayerBlock<Type>::forward() {
            Tensor<Type>& i_tensor = this->i_tensor;
            Tensor<Type>& o_tensor = this->o_tensor;
            auto* gathered_tensors = new Tensor<Type>[block.size()];

            for(size_t i = 0; i < block.size(); ++i) {
                Tensor<Type>& handle = i_tensor;
                for(auto& l: block[i]) {
                    l->fillInTensor(handle);
                    handle = (*l)();
                }
                gathered_tensors[i] = handle;
            }
            
            o_tensor = merge(gathered_tensors);
            delete[] gathered_tensors;
        }

        template <class Type>
        LayerBlock<Type>::~LayerBlock() {
        }

        template <class Type>
        void LayerBlock<Type>::summary() {
            std::cout << this << "In: (" << this->getInputShape().width << ", " << this->getInputShape().height << "," << this->getInputShape().depth << ")" 
            << " Out: (" << this->getOutputShape().width << ", " << this->getOutputShape().height << "," << this->getOutputShape().depth << ")" <<std::endl;            
            std::cout << "vvv" << std::endl;
           for(size_t i = 0; i < block.size(); ++i) {
                for(auto& l: block[i]) {
                    l->summary();
                }
                std::cout << "--\n";
            }                            
        }

        REGISTER_CLASS(LayerBlock,float);


    } // namespace Layers
} // namespace emptyNN