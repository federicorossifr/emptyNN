#include "emptyNN/Factory.hpp"
using namespace emptyNN;
int main() {
    // Layer<float> l({224,224,3},{128,128,3});
    // Layer<float> l2({224,224,3});
    // Layers::Impl::ConvCPUImpl<float> conv({6,6,1},{3,3,1},{{2,2,1},1,1});
    Layer<float>* l = Factory::Layers::Convolution<float>({6,6,1},{{2,2,1},1,1},Factory::Activations::Elu<float>(),CPU);
    (*l)();
}