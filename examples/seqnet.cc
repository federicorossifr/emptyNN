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
    seq.stackLayers({
        Convolution<floatx>({224,224,3},{{3,3,3},6,PaddingType::NONE}, nullptr,CPU)
    });
    floatx* in_tensor = new floatx[seq.getInputShape().size()];
    std::fill(in_tensor,in_tensor+seq.getInputShape().size(),0x01);
    seq.summary();
    std::cout << "Loaded model" << std::endl;
    floatx* out = seq.predict(in_tensor);
    seq.fit(in_tensor,out);

}

