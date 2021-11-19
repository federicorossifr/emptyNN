/*
emptyNN
Copyright (C) 2021 Federico Rossi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include <emptyNN/Factory.hpp>
#include <emptyNN/Sequential.hpp>
using namespace emptyNN;
using namespace Factory::Layers;
using namespace Factory::Activations;
#define RELU Factory::Activations::Elu<Type>(Type(1.))
#define SMAX Factory::Activations::Elu<Type>(Type(1.))
namespace emptyNN {
    namespace Models
    {
        template <class Type>
        Sequential<Type>* EfficientNetB0() {
            using floatx = Type;
            auto* s = new Sequential<Type>("EfficientNetB0");
            #define Squeeze(W,D,SQZ,DEV) \
                Multiply<floatx>({W,W,D},{W,W,D},{ \
                    {\
                        MaxPool<floatx>({W,W,D},{{W,W},1},nullptr,DEV),\
                        Convolution<floatx>({1,1,D},{{1,1,D},SQZ,1,PaddingType::SAME},nullptr,DEV),\
                        Convolution<floatx>({1,1,SQZ},{{1,1,SQZ},D,1,PaddingType::SAME},nullptr,DEV),\
                        Pad<floatx>({1,1,D},{(W)-1,(W)-1},DEV),\
                    },{}\
                },DEV)


            s->stackLayers({
                BatchNorm<floatx>({224,224,3},0,1,nullptr,CPU),
                Pad<floatx>({224,224,3},{1,1},CPU),
                Convolution<floatx>({225,225,3},{{3,3,3},32,2,PaddingType::ZERO},nullptr,CPU),        
                BatchNorm<floatx>({112,112,32},0,1,Elu<floatx>(1.),CPU),

                DWConvolution<floatx>({112,112,32},{{3,3,32},32,1,PaddingType::SAME,true},nullptr,CPU),
                BatchNorm<floatx>({112,112,32},0,1,Elu<floatx>(1.),CPU),

                Squeeze(112,32,8,CPU),

                Convolution<floatx>({112,112,32},{{1,1,32},16,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({112,112,16},0,1,Elu<floatx>(1.),CPU),


                Convolution<floatx>({112,112,16},{{1,1,16},96,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({112,112,96},0,1,Elu<floatx>(1.),CPU),
                Pad<floatx>({112,112,96},{1,1},CPU),
                DWConvolution<floatx>({113,113,96},{{2,2,96},96,2,PaddingType::ZERO,true},nullptr,CPU),
                BatchNorm<floatx>({56,56,96},0,1,Elu<floatx>(1.),CPU),

                Squeeze(56,96,4,CPU),  

                Convolution<floatx>({56,56,96},{{1,1,96},24,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({56,56,24},0,1,Elu<floatx>(1.),CPU),
                
                
                Add<floatx>({56,56,24},{56,56,24},{
                    {
                        Convolution<floatx>({56,56,24},{{1,1,24},144,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({56,56,144},0,1,Elu<floatx>(1.),CPU),
                        DWConvolution<floatx>({56,56,144},{{3,3,144},144,1,PaddingType::SAME,true},nullptr,CPU),                
                        BatchNorm<floatx>({56,56,144},0,1,Elu<floatx>(1.),CPU),   

                        Squeeze(56,144,6,CPU),

                        Convolution<floatx>({56,56,144},{{1,1,144},24,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({56,56,24},0,1,Elu<floatx>(1.),CPU),  
                    },
                    {}
                },CPU),

                
                Convolution<floatx>({56,56,24},{{1,1,24},144,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({56,56,144},0,1,Elu<floatx>(1.),CPU),  
                Pad<floatx>({56,56,144},{3,3},CPU),
                DWConvolution<floatx>({59,59,144},{{5,5,144},144,2,PaddingType::ZERO,true},nullptr,CPU),
                BatchNorm<floatx>({28,28,144},0,1,Elu<floatx>(1.),CPU),  

                Squeeze(28,144,6,CPU),

                Convolution<floatx>({28,28,144},{{1,1,144},40,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({28,28,40},0,1,Elu<floatx>(1.),CPU),
                
                
                Add<floatx>({28,28,40},{28,28,40},{
                    {
                        Convolution<floatx>({28,28,40},{{1,1,40},240,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({28,28,240},0,1,Elu<floatx>(1.),CPU),
                        DWConvolution<floatx>({28,28,240},{{3,3,240},240,1,PaddingType::SAME,true},nullptr,CPU),                
                        BatchNorm<floatx>({28,28,240},0,1,Elu<floatx>(1.),CPU),   

                        Squeeze(28,240,10,CPU),

                        Convolution<floatx>({28,28,240},{{1,1,240},40,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({28,28,40},0,1,Elu<floatx>(1.),CPU),  
                    },
                    {}
                },CPU),

                Convolution<floatx>({28,28,40},{{1,1,40},240,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({28,28,240},0,1,Elu<floatx>(1.),CPU),  
                Pad<floatx>({28,28,240},{1,1},CPU),
                DWConvolution<floatx>({29,29,240},{{3,3,240},240,2,PaddingType::ZERO,true},nullptr,CPU),
                BatchNorm<floatx>({14,14,240},0,1,Elu<floatx>(1.),CPU),  

                Squeeze(14,240,10,CPU),       

                Convolution<floatx>({14,14,240},{{1,1,240},80,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({14,14,80},0,1,Elu<floatx>(1.),CPU),
                
                
                Add<floatx>({14,14,80},{14,14,80},{
                    {
                        Convolution<floatx>({14,14,80},{{1,1,80},480,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({14,14,480},0,1,Elu<floatx>(1.),CPU),
                        DWConvolution<floatx>({14,14,480},{{3,3,480},480,1,PaddingType::SAME,true},nullptr,CPU),                
                        BatchNorm<floatx>({14,14,480},0,1,Elu<floatx>(1.),CPU),   

                        Squeeze(14,480,20,CPU),
                        
                        Convolution<floatx>({14,14,480},{{1,1,480},80,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({14,14,80},0,1,Elu<floatx>(1.),CPU),  
                    },
                    {}
                },CPU),    


                
                
                Add<floatx>({14,14,80},{14,14,80},{
                    {
                    
                        Convolution<floatx>({14,14,80},{{1,1,80},480,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({14,14,480},0,1,Elu<floatx>(1.),CPU),
                        DWConvolution<floatx>({14,14,480},{{3,3,480},480,1,PaddingType::SAME,true},nullptr,CPU),                
                        BatchNorm<floatx>({14,14,480},0,1,Elu<floatx>(1.),CPU),   

                        Squeeze(14,480,20,CPU),
                        
                        Convolution<floatx>({14,14,480},{{1,1,480},80,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({14,14,80},0,1,Elu<floatx>(1.),CPU),  
                    },
                    {}
                },CPU),            

                Convolution<floatx>({14,14,80},{{1,1,80},480,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({14,14,480},0,1,Elu<floatx>(1.),CPU),  
                DWConvolution<floatx>({14,14,480},{{5,5,480},480,1,PaddingType::SAME,true},nullptr,CPU),
                BatchNorm<floatx>({14,14,480},0,1,Elu<floatx>(1.),CPU),  

                Squeeze(14,480,20,CPU),

                Convolution<floatx>({14,14,480},{{1,1,480},112,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({14,14,112},0,1,Elu<floatx>(1.),CPU),              

                Add<floatx>({14,14,112},{14,14,112},{
                    {
                        Convolution<floatx>({14,14,112},{{1,1,672},672,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({14,14,672},0,1,Elu<floatx>(1.),CPU),                
                        DWConvolution<floatx>({14,14,672},{{5,5,672},672,1,PaddingType::SAME,true},nullptr,CPU),                
                        BatchNorm<floatx>({14,14,672},0,1,Elu<floatx>(1.),CPU),   

                        Squeeze(14,672,28,CPU),

                        Convolution<floatx>({14,14,672},{{1,1,672},112,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({14,14,112},0,1,Elu<floatx>(1.),CPU),  
                    },
                    {}
                },CPU), 

                Add<floatx>({14,14,112},{14,14,112},{
                    {
                        Convolution<floatx>({14,14,112},{{1,1,112},672,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({14,14,672},0,1,Elu<floatx>(1.),CPU),                
                        DWConvolution<floatx>({14,14,672},{{5,5,672},672,1,PaddingType::SAME,true},nullptr,CPU),                
                        BatchNorm<floatx>({14,14,672},0,1,Elu<floatx>(1.),CPU),   

                        Squeeze(14,672,28,CPU),

                        Convolution<floatx>({14,14,672},{{1,1,672},112,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({14,14,112},0,1,Elu<floatx>(1.),CPU),  
                    },
                    {}
                },CPU), 

                Convolution<floatx>({14,14,112},{{1,1,112},672,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({14,14,672},0,1,Elu<floatx>(1.),CPU),  
                Pad<floatx>({14,14,672},{3,3},CPU),
                DWConvolution<floatx>({17,17,672},{{5,5,672},672,2,PaddingType::ZERO,true},nullptr,CPU),
                BatchNorm<floatx>({7,7,672},0,1,Elu<floatx>(1.),CPU),  

                Squeeze(7,672,28,CPU),         
                
                Convolution<floatx>({7,7,672},{{1,1,672},192,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({7,7,192},0,1,Elu<floatx>(1.),CPU),              

                Add<floatx>({7,7,192},{7,7,192},{
                    {
                        Convolution<floatx>({7,7,192},{{1,1,192},1152,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({7,7,1152},0,1,Elu<floatx>(1.),CPU),                
                        DWConvolution<floatx>({7,7,1152},{{5,5,1152},1152,1,PaddingType::SAME,true},nullptr,CPU),                
                        BatchNorm<floatx>({7,7,1152},0,1,Elu<floatx>(1.),CPU),   

                        Squeeze(7,1152,48,CPU),

                        Convolution<floatx>({7,7,1152},{{1,1,1152},192,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({7,7,192},0,1,Elu<floatx>(1.),CPU),  
                    },
                    {}
                },CPU), 

                Add<floatx>({7,7,192},{7,7,192},{
                    {
                        Convolution<floatx>({7,7,192},{{1,1,192},1152,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({7,7,1152},0,1,Elu<floatx>(1.),CPU),                
                        DWConvolution<floatx>({7,7,1152},{{5,5,1152},1152,1,PaddingType::SAME,true},nullptr,CPU),                
                        BatchNorm<floatx>({7,7,1152},0,1,Elu<floatx>(1.),CPU),   

                        Squeeze(7,1152,48,CPU),

                        Convolution<floatx>({7,7,1152},{{1,1,1152},192,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({7,7,192},0,1,Elu<floatx>(1.),CPU),  
                    },
                    {}
                },CPU),

                Add<floatx>({7,7,192},{7,7,192},{
                    {
                        Convolution<floatx>({7,7,192},{{1,1,192},1152,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({7,7,1152},0,1,Elu<floatx>(1.),CPU),                
                        DWConvolution<floatx>({7,7,1152},{{5,5,1152},1152,1,PaddingType::SAME,true},nullptr,CPU),                
                        BatchNorm<floatx>({7,7,1152},0,1,Elu<floatx>(1.),CPU),   

                        Squeeze(7,1152,48,CPU),

                        Convolution<floatx>({7,7,1152},{{1,1,1152},192,1,PaddingType::ZERO},nullptr,CPU),
                        BatchNorm<floatx>({7,7,192},0,1,Elu<floatx>(1.),CPU),  
                    },
                    {}
                },CPU),      

                Convolution<floatx>({7,7,192},{{1,1,192},1152,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({7,7,1152},0,1,Elu<floatx>(1.),CPU),  
                DWConvolution<floatx>({7,7,1152},{{3,3,1152},1152,2,PaddingType::SAME,true},nullptr,CPU),
                BatchNorm<floatx>({7,7,1152},0,1,Elu<floatx>(1.),CPU),  

                Squeeze(7,1152,48,CPU),              

                Convolution<floatx>({7,7,1152},{{1,1,1152},320,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({7,7,320},0,1,Elu<floatx>(1.),CPU),        
                Convolution<floatx>({7,7,320},{{1,1,320},1280,1,PaddingType::ZERO},nullptr,CPU),
                BatchNorm<floatx>({7,7,1280},0,1,Elu<floatx>(1.),CPU),        
                MaxPool<floatx>({7,7,1280},{{7,7},1},nullptr,CPU),
                Dense<floatx>({1,1,1280},{1000,1,1},Elu<floatx>(1.),CPU)
            });
            return s;
        }

        template <class Type>
        Sequential<Type>* MobileNet() {
            auto* s = new Sequential<Type>("Mobilenet");
            s->stackLayers({
                Convolution<Type>({224,224,3},{{3,3,3},32,2, PaddingType::NONE},nullptr,CPU),
                Pad<Type>({111,111,32},{1,1,0},CPU),
                DWConvolution<Type>({112,112,32},{{3,3,32},32,1, PaddingType::SAME,true},nullptr,CPU),
                Convolution<Type>({112,112,32},{{1,1,32},64,1, PaddingType::SAME},nullptr,CPU),
                DWConvolution<Type>({112,112,64},{{3,3,64},64,2, PaddingType::NONE,true},nullptr,CPU),
                Pad<Type>({55,55,64},{1,1,0},CPU),

                Convolution<Type>({56,56,64},{{1,1,64},128,2, PaddingType::SAME},nullptr,CPU),
                DWConvolution<Type>({56,56,128},{{3,3,128},128,1, PaddingType::SAME,true},nullptr,CPU),
                Convolution<Type>({56,56,128},{{1,1,128},128,1, PaddingType::SAME},nullptr,CPU),
                DWConvolution<Type>({56,56,128},{{3,3,128},128,2, PaddingType::NONE,true},nullptr,CPU),
                Pad<Type>({27,27,128},{1,1,0},CPU),

                Convolution<Type>({28,28,128},{{1,1,128},256,1, PaddingType::SAME},nullptr,CPU),
                DWConvolution<Type>({28,28,256},{{3,3,256},256,2, PaddingType::SAME,true},nullptr,CPU),
                Convolution<Type>({28,28,256},{{1,1,256},256,1, PaddingType::SAME},nullptr,CPU),
                DWConvolution<Type>({28,28,256},{{3,3,256},256,2, PaddingType::NONE,true},nullptr,CPU),
                Pad<Type>({13,13,256},{1,1,0},CPU),

                Convolution<Type>({14,14,256},{{1,1,256},512,1, PaddingType::SAME},nullptr,CPU),

                DWConvolution<Type>({14,14,512},{{3,3,512},512,1, PaddingType::SAME,true},nullptr,CPU),
                Convolution<Type>({14,14,512},{{1,1,512},512,1, PaddingType::SAME},nullptr,CPU),
                DWConvolution<Type>({14,14,512},{{3,3,512},512,1, PaddingType::SAME,true},nullptr,CPU),
                Convolution<Type>({14,14,512},{{1,1,512},512,1, PaddingType::SAME},nullptr,CPU),
                DWConvolution<Type>({14,14,512},{{3,3,512},512,1, PaddingType::SAME,true},nullptr,CPU),
                Convolution<Type>({14,14,512},{{1,1,512},512,1, PaddingType::SAME},nullptr,CPU),
                DWConvolution<Type>({14,14,512},{{3,3,512},512,1, PaddingType::SAME,true},nullptr,CPU),
                Convolution<Type>({14,14,512},{{1,1,512},512,1, PaddingType::SAME},nullptr,CPU),
                DWConvolution<Type>({14,14,512},{{3,3,512},512,1, PaddingType::SAME,true},nullptr,CPU),
                Convolution<Type>({14,14,512},{{1,1,512},512,1, PaddingType::SAME},nullptr,CPU),

                DWConvolution<Type>({14,14,512},{{3,3,512},512,2, PaddingType::NONE,true},nullptr,CPU),
                Pad<Type>({6,6,512},{1,1,0},CPU),
                Convolution<Type>({7,7,512},{{1,1,512},1024,1, PaddingType::SAME},nullptr,CPU),
                DWConvolution<Type>({7,7,1024},{{3,3,1024},1024,2, PaddingType::SAME,true},nullptr,CPU),
                Convolution<Type>({7,7,1024},{{1,1,1024},1024,1, PaddingType::SAME},nullptr,CPU),

                MaxPool<Type>({7,7,1024},{{7,7},1},nullptr,CPU),
                Dense<Type>({1,1,1024},{1000,1,1},RELU,CPU)

            });
            return s;        
        }

        template <class Type>
        Sequential<Type>* LeNet5() {
            auto* s = new Sequential<Type>("LeNet5");
            s->stackLayers({
                Convolution<Type>({32,32,3},{{5,5,3},6,1},RELU,CPU), //450
                MaxPool<Type>({28,28,6},{{2,2},2},nullptr,CPU),
                Convolution<Type>({14,14,6},{{5,5,1},16,1},RELU,CPU), // 400
                MaxPool<Type>({10,10,16},{{2,2},2},nullptr,CPU),         
                Dense<Type>({5,5,16},{120,1,1},RELU,CPU), // 48000
                Dense<Type>({120,1,1},{84,1,1},RELU,CPU), // 10800
                Dense<Type>({84,1,1},{10,1,1},SMAX,CPU)   // 840    
            });
            return s;
        }

        template <class Type>
        Sequential<Type>* ResNet34(size_t num_classes) {
            auto* s = new Sequential<Type>("ResNet34");
            s->stackLayers({
                Factory::Layers::Convolution<Type>({224,224,3}, {{7,7,3}, 64, 2,PaddingType::SAME}, nullptr, CPU),
                Factory::Layers::MaxPool<Type>({224,224,64},{{1,1},2},nullptr,CPU),

                Factory::Layers::ResBlock<Type>({112,112,64},{56,56,64},{true,2,true},CPU),
                Factory::Layers::ResBlock<Type>({56,56,64},{56,56,64},{false,2,true},CPU),
                Factory::Layers::ResBlock<Type>({56,56,64},{56,56,64},{false,2,true},CPU),

                Factory::Layers::ResBlock<Type>({56,56,64},{28,28,128},{true,2,true},CPU),
                Factory::Layers::ResBlock<Type>({28,28,128},{28,28,128},{false,2,true},CPU),
                Factory::Layers::ResBlock<Type>({28,28,128},{28,28,128},{false,2,true},CPU),
                Factory::Layers::ResBlock<Type>({28,28,128},{28,28,128},{false,2,true},CPU),
                
                Factory::Layers::ResBlock<Type>({28,28,128},{14,14,256},{true,2,true},CPU),
                Factory::Layers::ResBlock<Type>({14,14,256},{14,14,256},{false,2,true},CPU),
                Factory::Layers::ResBlock<Type>({14,14,256},{14,14,256},{false,2,true},CPU),
                Factory::Layers::ResBlock<Type>({14,14,256},{14,14,256},{false,2,true},CPU),
                Factory::Layers::ResBlock<Type>({14,14,256},{14,14,256},{false,2,true},CPU),
                Factory::Layers::ResBlock<Type>({14,14,256},{14,14,256},{false,2,true},CPU),

                Factory::Layers::ResBlock<Type>({14,14,256},{7,7,512},{true,2,true},CPU),
                Factory::Layers::ResBlock<Type>({7,7,512},{7,7,512},{false,2,true},CPU),
                Factory::Layers::ResBlock<Type>({7,7,512},{7,7,512},{false,2,true},CPU),
                
                Factory::Layers::MaxPool<Type>({7,7,512},{{7,7},1},nullptr,CPU),
                Factory::Layers::Dense({1,1,512},{1,1,num_classes},RELU,CPU)
            });            
            return s;
        }

        template <class Type>
        Sequential<Type>* VGG16(size_t num_classes=1000, bool include_top=true, Shape input_shape={224,224,3}) {
            auto* s = new Sequential<Type>("VGG16");
            Layer<Type>* sc;
            Shape tmp;
            s->stackLayers({
                
                //conv_1
                Factory::Layers::Convolution<Type>(input_shape, {{3,3,3}, 64, 1,PaddingType::SAME}, RELU, CPU),
                Factory::Layers::Convolution<Type>({input_shape.width,input_shape.height,64}, {{3,3,64}, 64, 1,PaddingType::SAME}, RELU, CPU),
                sc = Factory::Layers::MaxPool<Type>({input_shape.width,input_shape.height,64},{{2,2},2},nullptr,CPU),

                //conv_2
                Factory::Layers::Convolution<Type>(tmp=sc->getOutputShape(), {{3,3,64}, 128, 1,PaddingType::SAME}, RELU, CPU),
                Factory::Layers::Convolution<Type>({tmp.width,tmp.height,128}, {{3,3,128}, 128, 1,PaddingType::SAME}, RELU, CPU),
                sc = Factory::Layers::MaxPool<Type>({tmp.width,tmp.height,128},{{1,1},2},nullptr,CPU),

                //conv_3
                Factory::Layers::Convolution<Type>(tmp=sc->getOutputShape(), {{3,3,128}, 256, 1,PaddingType::SAME}, RELU, CPU),
                Factory::Layers::Convolution<Type>({tmp.width,tmp.height,256}, {{3,3,256}, 256, 1,PaddingType::SAME}, RELU, CPU),                
                Factory::Layers::Convolution<Type>({tmp.width,tmp.height,256}, {{3,3,256}, 256, 1,PaddingType::SAME}, RELU, CPU),                
                sc = Factory::Layers::MaxPool<Type>({tmp.width,tmp.height,256},{{1,1},2},nullptr,CPU),

                //conv_4
                Factory::Layers::Convolution<Type>(tmp=sc->getOutputShape(), {{3,3,256}, 512, 1,PaddingType::SAME}, RELU, CPU),
                Factory::Layers::Convolution<Type>({tmp.width,tmp.height,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),                
                Factory::Layers::Convolution<Type>({tmp.width,tmp.height,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),                
                sc = Factory::Layers::MaxPool<Type>({tmp.width,tmp.height,512},{{1,1},2},nullptr,CPU),

                //conv_5
                Factory::Layers::Convolution<Type>(tmp=sc->getOutputShape(), {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),
                Factory::Layers::Convolution<Type>({tmp.width,tmp.height,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),                
                Factory::Layers::Convolution<Type>({tmp.width,tmp.height,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),                
                sc = Factory::Layers::MaxPool<Type>({tmp.width,tmp.height,512},{{1,1},2},nullptr,CPU)
            });
            if(include_top) {
                s->stackLayers({
                    Factory::Layers::Dense(sc->getOutputShape(),{1,1,4096},RELU,CPU),
                    Factory::Layers::Dense({1,1,4096},{1,1,4096},RELU,CPU),
                    Factory::Layers::Dense({1,1,4096},{1,1,num_classes},SMAX,CPU),
                });
            }

            return s;
        }

        template <class Type>
        Sequential<Type>* SSD300() {
            using floatx = Type;
            auto* s = new Sequential<Type>("SSD300");
            s->stackLayers({
                //conv_1
                Convolution<floatx>({300,300,3}, {{3,3,3}, 64, 1,PaddingType::SAME}, RELU, CPU),
                Convolution<floatx>({300,300,64}, {{3,3,64}, 64, 1,PaddingType::SAME}, RELU, CPU),
                MaxPool<floatx>({300,300,64},{{2,2},2},nullptr,CPU),

                //conv_2
                Convolution<floatx>({150,150,64}, {{3,3,64}, 128, 1,PaddingType::SAME}, RELU, CPU),
                Convolution<floatx>({150,150,128}, {{3,3,128}, 128, 1,PaddingType::SAME}, RELU, CPU),
                MaxPool<floatx>({150,150,128},{{1,1},2},nullptr,CPU),

                //conv_3
                Convolution<floatx>({75,75,128}, {{3,3,128}, 256, 1,PaddingType::SAME}, RELU, CPU),
                Convolution<floatx>({75,75,256}, {{3,3,256}, 256, 1,PaddingType::SAME}, RELU, CPU),
                Convolution<floatx>({75,75,256}, {{3,3,256}, 256, 1,PaddingType::SAME}, RELU, CPU),
                MaxPool<floatx>({75,75,256},{{1,1},2},nullptr,CPU),

                //conv_4
                Convolution<floatx>({38,38,256}, {{3,3,256}, 512, 1,PaddingType::SAME}, RELU, CPU),
                Convolution<floatx>({38,38,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),
                Convolution<floatx>({38,38,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),
                Concat<floatx>({38,38,512},{218300,1,1},{
                    {   // Out: 144000
                        Convolution<floatx>({38,38,512}, {{3,3,512}, 4*(21+4), 1,PaddingType::SAME}, RELU, CPU),Flatten<floatx>({38,38,4*(21+4)},CPU)
                    },
                    {
                        MaxPool<floatx>({38,38,512},{{1,1},2},nullptr,CPU),

                        //conv_5
                        Convolution<floatx>({19,19,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),
                        Convolution<floatx>({19,19,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),
                        Convolution<floatx>({19,19,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),

                        //conv_6      
                        Convolution<floatx>({19,19,512}, {{3,3,512}, 1024, 1,PaddingType::SAME}, RELU, CPU),
                        
                        //conv_7
                        Convolution<floatx>({19,19,1024}, {{1,1,1024}, 1024, 1,PaddingType::SAME}, RELU, CPU),
                        Concat<floatx>({19,19,1024},{73900,1,1},{ // Out: 73900
                            {   // Out: 54150
                                Convolution<floatx>({19,19,1024}, {{3,3,1024}, 6*(21+4), 1,PaddingType::SAME}, RELU, CPU),Flatten<floatx>({19,19,6*(21+4)},CPU)
                            },
                            {
                                // conv_8_2
                                Convolution<floatx>({19,19,1024}, {{1,1,1024}, 256, 1,PaddingType::SAME}, RELU, CPU),
                                MaxPool<floatx>({19,19,256},{{1,1},2},nullptr,CPU),
                                Convolution<floatx>({10,10,256}, {{1,1,256}, 512, 1,PaddingType::SAME}, RELU, CPU),
                                Concat<floatx>({10,10,512},{19750,1,1},{ // Out: 19750
                                    {  // Out: 15000
                                    Convolution<floatx>({10,10,512}, {{3,3,512}, 6*(21+4), 1,PaddingType::SAME}, RELU, CPU),Flatten<floatx>({10,10,6*(21+4)},CPU)
                                    },
                                    {
                                        // conv_9_2
                                        Convolution<floatx>({10,10,512}, {{1,1,1024}, 128, 1,PaddingType::SAME}, RELU, CPU),
                                        MaxPool<floatx>({10,10,128},{{1,1},2},nullptr,CPU),
                                        Convolution<floatx>({5,5,128}, {{1,1,128}, 256, 1,PaddingType::SAME}, RELU, CPU),
                                        Concat<floatx>({5,5,256},{4750,1,1},{ // Out: 4750
                                            {   // Out: 3750
                                                Convolution<floatx>({5,5,256}, {{3,3,256}, 6*(21+4), 1,PaddingType::SAME}, RELU, CPU),Flatten<floatx>({5,5,6*(21+4)},CPU)
                                            },
                                            { 
                                                Convolution<floatx>({5,5,256}, {{1,1,256}, 128, 1,PaddingType::SAME}, RELU, CPU),
                                                MaxPool<floatx>({5,5,128},{{1,1},2},nullptr,CPU),
                                                Convolution<floatx>({3,3,128}, {{1,1,128}, 256, 1,PaddingType::SAME}, RELU, CPU),
                                                Concat<floatx>({3,3,256},{1000,1,1},{ // Out 1000
                                                    {   // Out: 900
                                                        Convolution<floatx>({3,3,256}, {{3,3,256}, 4*(21+4), 1,PaddingType::SAME}, RELU, CPU),Flatten<floatx>({3,3,4*(21+4)},CPU)
                                                    },
                                                    {
                                                        Convolution<floatx>({3,3,256}, {{1,1,256}, 128, 1,PaddingType::SAME}, RELU, CPU),
                                                        MaxPool<floatx>({3,3,128},{{2,2},2},nullptr,CPU),
                                                        Convolution<floatx>({1,1,128}, {{1,1,128}, 256, 1,PaddingType::SAME}, RELU, CPU),
                                                        Convolution<floatx>({1,1,256}, {{3,3,256}, 4*(21+4), 1,PaddingType::SAME}, RELU, CPU),
                                                        Flatten<floatx>({1,1,4*(21+4)},CPU) // Out: 100
                                                    }
                                                },CPU)
                                            }
                                        },CPU)
                                    }
                                },CPU)  
                            }
                        },CPU)
                    }
                },CPU)
            });
            return s;
        }

    } // namespace Models
} // namespace emptyNN
