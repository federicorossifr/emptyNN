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
#include <emptyNN/layers/core/Concat_cpu_impl.hpp>
#include <emptyNN/layers/core/Conv_cpu_impl.hpp>
#include <emptyNN/activations/Elu.hpp>

namespace emptyNN {
    namespace Layers {
        namespace Impl {

            template <class Type>
            ConcatCPUImpl<Type>::ConcatCPUImpl(Shape in, Shape out, std::vector<std::vector<std::unique_ptr<Layer<Type>>>>&& _block): LayerBlock<Type>(in, out, std::move(_block)) {
                // ToDo: sanity checks on output 
                // Output should be sum of all the getOutputShape().size()
                // from layers
                // Maybe make a constructor without "out" parameter
                // and infer directly the output
            }

            template <class Type>
            Tensor<Type>& ConcatCPUImpl<Type>::merge(Tensor<Type> tensors[]) {

                Tensor<Type> o_tensor = this->o_tensor;
                auto o_it = o_tensor.begin();

                auto& blocks = this->block;
                size_t n_tensors = this->block.size();

                #pragma omp parallel for
                for(size_t i = 0; i < n_tensors; ++i ) { // In-depth is <= out depth
                    Shape out_shape_i = blocks[i].back()->getOutputShape();
                    size_t area_i = out_shape_i.size();
                    Tensor<Type> tensor_i = tensors[i];
                    std::copy(tensor_i.begin(),tensor_i.end(),o_it);
                    o_it+=area_i;
                }
                
                return this->o_tensor;
            }   

            REGISTER_CLASS(ConcatCPUImpl,float);

        } // namespace Impl
    } // namespace Layers
} // namespace emptyNN
