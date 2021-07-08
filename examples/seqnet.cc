#include "emptyNN/ModelFactory.hpp"
#include "emptyNN/Sequential.hpp"
#include <iostream>
using namespace emptyNN;
int main() {
    //Sequential<float>* s = Models::VGG16<float>(100,true,{300,300,3});    
    Sequential<float> ccnet("ConcatNet");
    ccnet.stackLayers({
        Factory::Layers::Dense<float>({100,1,1},{50,1,1},nullptr,CPU),
        Factory::Layers::Concat<float>({50,1,1},{20,1,1},{
            {Factory::Layers::Dense<float>({50,1,1},{10,1,1},nullptr,CPU)},
            {Factory::Layers::Dense<float>({50,1,1},{10,1,1},nullptr,CPU)}
        },CPU)
    });
    ccnet.summary();
    Layer<float>* shortcut;
    
    float* in_tensor = new float[ccnet.getInputShape().size()];
    std::fill(in_tensor,in_tensor+ccnet.getInputShape().size(),0x1);
    float* out_tensor = ccnet.predict(in_tensor);
    delete[] in_tensor;
    delete[] out_tensor;
}