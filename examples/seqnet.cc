#include "emptyNN/Factory.hpp"
#include "emptyNN/Sequential.hpp"
#include <iostream>
using namespace emptyNN;
int main() {
    Sequential<float> s("Example");
    
    
    #define ELU Factory::Activations::Elu<float>(1.)
    Layer<float>* shortcut;
    s.stackLayers({
        Factory::Layers::Convolution<float>({224,224,3}, {{11,11,3}, 96, 4,PaddingType::ZERO}, ELU, CPU),
        Factory::Layers::MaxPool<float>({54,54,96},{{3,3},2},nullptr,CPU),
        Factory::Layers::Convolution<float>({26,26,96}, {{5,5,96}, 256, 1,PaddingType::SAME}, ELU, CPU),
        shortcut = Factory::Layers::MaxPool<float>({26,26,256},{{3,3},2},nullptr,CPU),
        Factory::Layers::Convolution<float>({12,12,256}, {{5,5,256}, 384, 1,PaddingType::SAME}, ELU, CPU),
        Factory::Layers::Convolution<float>({12,12,384}, {{5,5,384}, 384, 1,PaddingType::SAME}, ELU, CPU),
        Factory::Layers::Convolution<float>({12,12,384}, {{5,5,384}, 256, 1,PaddingType::SAME}, ELU, CPU),
        Factory::Layers::MaxPool<float>({12,12,256},{{3,3},2},nullptr,CPU),
        Factory::Layers::Dense<float>({5,5,256},{4096,1,1},ELU,CPU),
        Factory::Layers::Dense<float>({4096,1,1},{4096,1,1},ELU,CPU),
        Factory::Layers::Dense<float>({4096,1,1},{1000,1,1},ELU,CPU)
    });
    float* in_tensor = new float[s.getInputShape().size()];
    //std::cout << "Predicting" << std::endl;
    float* out_tensor = s.predict(in_tensor);
    //delete[] in_tensor;
    // delete[] out_tensor;
}