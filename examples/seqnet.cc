#include "emptyNN/ModelFactory.hpp"
#include "emptyNN/Sequential.hpp"
#include <iostream>
#include <new>
using namespace emptyNN;
using namespace Factory::Layers;
int main() {
    Sequential<float> ccnet("Mobilenet");
    ccnet.stackLayers({
        Convolution<float>({224,224,3},{{3,3,3},32,2, PaddingType::NONE},nullptr,CPU),
        Pad<float>({111,111,32},{1,1,0},CPU),
        DWConvolution<float>({112,112,32},{{3,3,32},32,1, PaddingType::SAME,true},nullptr,CPU),
        Convolution<float>({112,112,32},{{1,1,32},64,1, PaddingType::SAME},nullptr,CPU),
        DWConvolution<float>({112,112,64},{{3,3,64},64,2, PaddingType::NONE,true},nullptr,CPU),
        Pad<float>({55,55,64},{1,1,0},CPU),

        Convolution<float>({56,56,64},{{1,1,64},128,2, PaddingType::SAME},nullptr,CPU),
        DWConvolution<float>({56,56,128},{{3,3,128},128,1, PaddingType::SAME,true},nullptr,CPU),
        Convolution<float>({56,56,128},{{1,1,128},128,1, PaddingType::SAME},nullptr,CPU),
        DWConvolution<float>({56,56,128},{{3,3,128},128,2, PaddingType::NONE,true},nullptr,CPU),
        Pad<float>({27,27,128},{1,1,0},CPU),

        Convolution<float>({28,28,128},{{1,1,128},256,1, PaddingType::SAME},nullptr,CPU),
        DWConvolution<float>({28,28,256},{{3,3,256},256,2, PaddingType::SAME,true},nullptr,CPU),
        Convolution<float>({28,28,256},{{1,1,256},256,1, PaddingType::SAME},nullptr,CPU),
        DWConvolution<float>({28,28,256},{{3,3,256},256,2, PaddingType::NONE,true},nullptr,CPU),
        Pad<float>({13,13,256},{1,1,0},CPU),

        Convolution<float>({14,14,256},{{1,1,256},512,1, PaddingType::SAME},nullptr,CPU), 

        DWConvolution<float>({14,14,512},{{3,3,512},512,1, PaddingType::SAME,true},nullptr,CPU),
        Convolution<float>({14,14,512},{{1,1,512},512,1, PaddingType::SAME},nullptr,CPU),
        DWConvolution<float>({14,14,512},{{3,3,512},512,1, PaddingType::SAME,true},nullptr,CPU),
        Convolution<float>({14,14,512},{{1,1,512},512,1, PaddingType::SAME},nullptr,CPU),
        DWConvolution<float>({14,14,512},{{3,3,512},512,1, PaddingType::SAME,true},nullptr,CPU),
        Convolution<float>({14,14,512},{{1,1,512},512,1, PaddingType::SAME},nullptr,CPU),
        DWConvolution<float>({14,14,512},{{3,3,512},512,1, PaddingType::SAME,true},nullptr,CPU),
        Convolution<float>({14,14,512},{{1,1,512},512,1, PaddingType::SAME},nullptr,CPU),
        DWConvolution<float>({14,14,512},{{3,3,512},512,1, PaddingType::SAME,true},nullptr,CPU),
        Convolution<float>({14,14,512},{{1,1,512},512,1, PaddingType::SAME},nullptr,CPU),

        DWConvolution<float>({14,14,512},{{3,3,512},512,2, PaddingType::NONE,true},nullptr,CPU),
        Pad<float>({6,6,512},{1,1,0},CPU),
        Convolution<float>({7,7,512},{{1,1,512},1024,1, PaddingType::SAME},nullptr,CPU),
        DWConvolution<float>({7,7,1024},{{3,3,1024},1024,2, PaddingType::SAME,true},nullptr,CPU),
        Convolution<float>({7,7,1024},{{1,1,1024},1024,1, PaddingType::SAME},nullptr,CPU),

        MaxPool<float>({7,7,1024},{{7,7},1},nullptr,CPU),
        Dense<float>({1,1,1024},{1000,1,1},Factory::Activations::Elu(1.f),CPU)

    });
    ccnet.summary();
    Layer<float>* shortcut;
    
    float*  in_tensor = new float[ccnet.getInputShape().size()];
    std::fill(in_tensor,in_tensor+ccnet.getInputShape().size(),0x1);
    float* out_tensor = ccnet.predict(in_tensor);
    delete[] in_tensor;
    delete[] out_tensor;
}

