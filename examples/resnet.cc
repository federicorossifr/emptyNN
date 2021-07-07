#include "emptyNN/Factory.hpp"
#include "emptyNN/Sequential.hpp"
#include <iostream>
using namespace emptyNN;
int main() {
    using floatx = float;
    Sequential<floatx> s("Example");
    
    #define ELU Factory::Activations::Elu<floatx>(1.)
    s.stackLayers({
        Factory::Layers::Convolution<floatx>({224,224,3}, {{7,7,3}, 64, 2,PaddingType::SAME}, nullptr, CPU),
        Factory::Layers::MaxPool<floatx>({224,224,64},{{1,1},2},nullptr,CPU),

        Factory::Layers::ResBlock<floatx>({112,112,64},{56,56,64},{true,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({56,56,64},{56,56,64},{false,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({56,56,64},{56,56,64},{false,2,true},CPU),

        Factory::Layers::ResBlock<floatx>({56,56,64},{28,28,128},{true,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({28,28,128},{28,28,128},{false,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({28,28,128},{28,28,128},{false,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({28,28,128},{28,28,128},{false,2,true},CPU),
        
        Factory::Layers::ResBlock<floatx>({28,28,128},{14,14,256},{true,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({14,14,256},{14,14,256},{false,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({14,14,256},{14,14,256},{false,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({14,14,256},{14,14,256},{false,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({14,14,256},{14,14,256},{false,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({14,14,256},{14,14,256},{false,2,true},CPU),

        Factory::Layers::ResBlock<floatx>({14,14,256},{7,7,512},{true,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({7,7,512},{7,7,512},{false,2,true},CPU),
        Factory::Layers::ResBlock<floatx>({7,7,512},{7,7,512},{false,2,true},CPU),
        
        Factory::Layers::MaxPool<floatx>({7,7,512},{{7,7},1},nullptr,CPU),
        Factory::Layers::Dense({1,1,512},{1,1,1000},ELU,CPU)

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