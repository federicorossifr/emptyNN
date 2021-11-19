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
#include "emptyNN/Factory.hpp"
#include "emptyNN/Sequential.hpp"
#include "emptyNN/io/Serializer.hpp"
#include <iostream>
#include <new>
using namespace emptyNN;
using namespace Factory::Layers;
using namespace Factory::Activations;
int main() {
    using floatx = float;
    Random::globalSeed = 2021;
    Sequential<floatx> seq("Effnet");

    std::vector<std::unique_ptr<Layer<floatx>>> layers;
    layers.push_back(std::move(Convolution<floatx>({224,224,3},{{3,3,3},6,2,PaddingType::NONE}, nullptr,CPU)));

    seq.stackLayers(std::move(layers));
    auto in_tensor = Tensor<floatx>(seq.getInputShape().size());
    std::fill(in_tensor.begin(),in_tensor.end(),0x01);
    seq.summary();
    std::cout << "Loaded model" << std::endl;
    auto out = seq.predict(in_tensor);
    seq.fit(in_tensor,out);

}

