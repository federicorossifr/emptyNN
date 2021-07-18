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
#include "emptyNN/utils/chrono_utils.hpp"
#include <iostream>
#include <new>
using namespace emptyNN;




int main() {
    #ifdef TYPE
    using floatx = TYPE;
    #else
    using floatx = float;
    #endif
    Sequential<floatx>* s = Models::EfficientNetB0<floatx>();

    floatx* in_tensor = new floatx[s->getInputShape().size()];
    std::fill(in_tensor,in_tensor+s->getInputShape().size(),0x01);

    size_t runs = 1e2;
    std::cout << "Predicting " << runs << " runs" << std::endl;
    double duration_ns = 0.;
    floatx* out_tensor;
    for(size_t i = 0; i < runs; ++i) {
        chronoIt([&in_tensor,&s]() {
          s->predict(in_tensor);
        }, [&duration_ns](double elapsed) {
            std::cout << elapsed/1e9 << std::endl;
            duration_ns+=elapsed;
        });
    }
    std::cout << "\nPredicted " << runs << " frames in: " << duration_ns*1e-9 << "s" << std::endl;
    std::cout << "\nAverage FPS: " << runs/(duration_ns/1e9) << std::endl;
    delete[] in_tensor;
    delete[] out_tensor;
    return 0;
}