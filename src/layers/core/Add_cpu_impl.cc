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
#include <emptyNN/layers/core/Add_cpu_impl.hpp>
#include <emptyNN/activations/Elu.hpp>

namespace emptyNN {
    namespace Layers {
        namespace Impl {

            template <class Type>
            AddCPUImpl<Type>::AddCPUImpl(Shape in, Shape out, std::vector<std::vector<Layer<Type>*>> _block): LayerBlock<Type>(in, out, _block) {
                // ToDo: sanity checks on output 
                // Output size should be the same as each output size
                std::fill(this->o_tensor,this->o_tensor+out.size(),0x0);
            }

            template <class Type>
            Type* AddCPUImpl<Type>::merge(Type* tensors[]) {
                Type* o_tensor = this->o_tensor;
                Shape out = this->o_shape;
                size_t n_tensors = this->block.size();
                #pragma omp parallel for
                for(size_t i = 0; i < out.size(); ++i ) {
                    o_tensor[i] = 0;
                              
                    for(size_t j = 0; j < n_tensors; ++j)
                        o_tensor[i]+=tensors[j][i];
                }
                return o_tensor;
            }   

            REGISTER_CLASS(AddCPUImpl,float);

        } // namespace Impl
    } // namespace Layers
} // namespace emptyNN
