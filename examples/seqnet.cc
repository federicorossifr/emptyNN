#include "emptyNN/Factory.hpp"
#include "emptyNN/Sequential.hpp"
using namespace emptyNN;
int main() {
    Sequential<float> s("Example");
    s.stackLayer( Factory::Layers::Convolution<float>({224,224,3}, {{5,5,3}, 16, 1}, Factory::Activations::Elu<float>(1.), CPU));
    s.stackLayer( Factory::Layers::Convolution<float>({220,220,16}, {{5,5,16}, 16, 1}, Factory::Activations::Elu<float>(1.), CPU));
    s.stackLayer( Factory::Layers::Convolution<float>({216,216,16}, {{5,5,16}, 16, 1}, Factory::Activations::Elu<float>(1.), CPU));
    float* in_tensor = new float[s.getInputShape().size()];
    float* out_tensor = s.predict(in_tensor);
    delete[] in_tensor;
    delete[] out_tensor;
}