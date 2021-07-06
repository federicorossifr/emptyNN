#include "emptyNN/Factory.hpp"
#include "emptyNN/Sequential.hpp"
#include <iostream>
using namespace emptyNN;
int main() {
    Sequential<float> s("Example");
    
    
    #define ELU Factory::Activations::Elu<float>(1.)
    s.stackLayers({
        Factory::Layers::Convolution<float>({224,224,3}, {{7,7,3}, 64, 2,PaddingType::SAME}, nullptr, CPU),
        Factory::Layers::MaxPool<float>({224,224,64},{{1,1},2},nullptr,CPU),

        Factory::Layers::ResBlock<float>({112,112,64},{56,56,64},{true,2,true},CPU),
        Factory::Layers::ResBlock<float>({56,56,64},{56,56,64},{false,2,true},CPU),
        Factory::Layers::ResBlock<float>({56,56,64},{56,56,64},{false,2,true},CPU),

        Factory::Layers::ResBlock<float>({56,56,64},{28,28,128},{true,2,true},CPU),
        Factory::Layers::ResBlock<float>({28,28,128},{28,28,128},{false,2,true},CPU),
        Factory::Layers::ResBlock<float>({28,28,128},{28,28,128},{false,2,true},CPU),
        Factory::Layers::ResBlock<float>({28,28,128},{28,28,128},{false,2,true},CPU),
        
        Factory::Layers::ResBlock<float>({28,28,128},{14,14,256},{true,2,true},CPU),
        Factory::Layers::ResBlock<float>({14,14,256},{14,14,256},{false,2,true},CPU),
        Factory::Layers::ResBlock<float>({14,14,256},{14,14,256},{false,2,true},CPU),
        Factory::Layers::ResBlock<float>({14,14,256},{14,14,256},{false,2,true},CPU),
        Factory::Layers::ResBlock<float>({14,14,256},{14,14,256},{false,2,true},CPU),
        Factory::Layers::ResBlock<float>({14,14,256},{14,14,256},{false,2,true},CPU),

        Factory::Layers::ResBlock<float>({14,14,256},{7,7,512},{true,2,true},CPU),
        Factory::Layers::ResBlock<float>({7,7,512},{7,7,512},{false,2,true},CPU),
        Factory::Layers::ResBlock<float>({7,7,512},{7,7,512},{false,2,true},CPU),
        
        Factory::Layers::MaxPool<float>({7,7,512},{{7,7},1},nullptr,CPU),
        Factory::Layers::Dense({1,1,512},{1,1,1000},ELU,CPU)

    });
    float* in_tensor = new float[s.getInputShape().size()];
    std::fill(in_tensor,in_tensor+s.getInputShape().size(),0x01);
    //std::cout << "Predicting" << std::endl;
    std::cout << "Predicting" << std::endl;
    float* out_tensor = s.predict(in_tensor);
    std::cout << "Done" << std::endl;
    delete[] in_tensor;
    delete[] out_tensor;
}