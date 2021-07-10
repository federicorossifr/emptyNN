#include "emptyNN/ModelFactory.hpp"
#include <iostream>
#include <new>
using namespace emptyNN;
int main() {
    using floatx = float;
    
    Sequential<floatx> s("RVV");
    s.stackLayers({
        Convolution<floatx>({224,224,3},{ {3,3,3},3,1,PaddingType::SAME}, nullptr, CPU_RVV)
    });
    floatx* in_tensor = new floatx[s.getInputShape().size()];
    std::fill(in_tensor,in_tensor+s.getInputShape().size(),0x01);
    //std::cout << "Predicting" << std::endl;
    std::cout << "Predicting" << std::endl;
    floatx* out_tensor = s.predict(in_tensor);
    std::cout << "Done" << std::endl;
    delete[] in_tensor;
    delete[] out_tensor;
}