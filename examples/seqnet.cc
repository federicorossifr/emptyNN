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
#include "emptyNN/Factory.hpp"
#include "emptyNN/Sequential.hpp"
#include "emptyNN/io/Serializer.hpp"
#include <iostream>
#include <new>
using namespace emptyNN;
using namespace Factory::Layers;
using namespace Factory::Activations;
int main() {
    using floatx = float;
    Random::globalSeed = 2021;
    Sequential<floatx> seq("Effnet");
    //seq = Models::VGG16<floatx>();
    #define _ELU = Factory::Activations::Elu<floatx>(1.);
    #define Squeeze(W,D,SQZ,DEV) \
        Add<floatx>({W,W,D},{W,W,D},{ \
            {\
                MaxPool<floatx>({W,W,D},{{W,W},1},nullptr,DEV),\
                Convolution<floatx>({1,1,D},{{1,1,D},SQZ,1,PaddingType::SAME},nullptr,DEV),\
                Convolution<floatx>({1,1,SQZ},{{1,1,SQZ},D,1,PaddingType::SAME},nullptr,DEV),\
                Pad<floatx>({1,1,D},{W-1,W-1},DEV),\
            },{}\
        },DEV)


    seq.stackLayers({
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

    //io::Serializer<floatx> ser("test_fp.bin");
    seq.summary();
    //ser.dumpBinaryWeights(seq);
    std::cout << "Loaded model" << std::endl;
}

