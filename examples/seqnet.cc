#include "emptyNN/ModelFactory.hpp"
#include "emptyNN/Sequential.hpp"
#include <iostream>
#include <new>
using namespace emptyNN;
using namespace Factory::Layers;
int main() {
    Sequential<float>* ccnet;
    ccnet = Models::SSD300<float>();
    ccnet->summary();
    Layer<float>* shortcut;
    
    float*  in_tensor = new float[ccnet->getInputShape().size()];
    std::fill(in_tensor,in_tensor+ccnet->getInputShape().size(),0x1);
    float* out_tensor = ccnet->predict(in_tensor);
    delete[] in_tensor;
    delete[] out_tensor;
}

