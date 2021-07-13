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
    Sequential<floatx>* s = Models::LeNet5<floatx>();

    floatx* in_tensor = new floatx[s->getInputShape().size()];
    std::fill(in_tensor,in_tensor+s->getInputShape().size(),0x01);

    size_t runs = 1e3;
    std::cout << "Predicting " << runs << " runs" << std::endl;
    double duration_ns = 0.;
    floatx* out_tensor;
    for(size_t i = 0; i < runs; ++i) {
        chronoIt([&in_tensor,&s]() {
          s->predict(in_tensor);  
        }, [&duration_ns](double elapsed) {
            duration_ns+=elapsed;
        });
    }
    std::cout << "\nPredicted " << runs << " frames in: " << duration_ns*1e-9 << "s" << std::endl;
    std::cout << "\nAverage FPS: " << runs/(duration_ns/1e9) << std::endl;
    delete[] in_tensor;
    delete[] out_tensor;
}