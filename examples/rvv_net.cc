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
#include "emptyNN/ModelFactory.hpp"
#include "emptyNN/io/BinarySerializer.hpp"
#include <iostream>
#include <new>
using namespace emptyNN;
int main() {
    using floatx = Posit8_0;
    
    Sequential<floatx> s("RVV");
    s.stackLayers({
        Convolution<floatx>({224,224,3},{ {3,3,3},3,1,PaddingType::SAME}, nullptr, CPU_RVV)
    });
    io::BinarySerializer<floatx> ser("test.bin");
    ser.serialize(&s);


    /*floatx* in_tensor = new floatx[s.getInputShape().size()];
    std::fill(in_tensor,in_tensor+s.getInputShape().size(),0x01);
    //std::cout << "Predicting" << std::endl;
    std::cout << "Predicting" << std::endl;
    floatx* out_tensor = s.predict(in_tensor);
    std::cout << "Done" << std::endl;
    delete[] in_tensor;
    delete[] out_tensor;*/
}