#include <emptyNN/Factory.hpp>
#include <emptyNN/Sequential.hpp>
using namespace emptyNN;
using namespace Factory::Layers;
#define RELU Factory::Activations::Elu<Type>(1.)
#define SMAX Factory::Activations::Elu<Type>(1.)
namespace emptyNN
{
    namespace Models
    {
        template <class Type>
        Sequential<Type>* ResNet34(size_t num_classes) {
            Sequential<Type>* s = new Sequential<Type>("ResNet34");
            #define ELU Factory::Activations::Elu<Type>(1.)
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
                Factory::Layers::Dense({1,1,512},{1,1,num_classes},ELU,CPU)
            });            
            return s;
        }

        template <class Type>
        Sequential<Type>* VGG16(size_t num_classes=1000, bool include_top=true, Shape input_shape={224,224,3}) {
            Sequential<Type>* s = new Sequential<Type>("VGG16");
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
            Sequential<Type>* s = new Sequential<Type>("SSD300");            
            s->stackLayers({
                //conv_1
                Convolution<float>({300,300,3}, {{3,3,3}, 64, 1,PaddingType::SAME}, RELU, CPU),
                Convolution<float>({300,300,64}, {{3,3,64}, 64, 1,PaddingType::SAME}, RELU, CPU),
                MaxPool<float>({300,300,64},{{2,2},2},nullptr,CPU),

                //conv_2
                Convolution<float>({150,150,64}, {{3,3,64}, 128, 1,PaddingType::SAME}, RELU, CPU),
                Convolution<float>({150,150,128}, {{3,3,128}, 128, 1,PaddingType::SAME}, RELU, CPU),
                MaxPool<float>({150,150,128},{{1,1},2},nullptr,CPU),

                //conv_3
                Convolution<float>({75,75,128}, {{3,3,128}, 256, 1,PaddingType::SAME}, RELU, CPU),
                Convolution<float>({75,75,256}, {{3,3,256}, 256, 1,PaddingType::SAME}, RELU, CPU),                
                Convolution<float>({75,75,256}, {{3,3,256}, 256, 1,PaddingType::SAME}, RELU, CPU),                
                MaxPool<float>({75,75,256},{{1,1},2},nullptr,CPU),

                //conv_4
                Convolution<float>({38,38,256}, {{3,3,256}, 512, 1,PaddingType::SAME}, RELU, CPU),
                Convolution<float>({38,38,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),                
                Convolution<float>({38,38,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),    
                Concat<float>({38,38,512},{218300,1,1},{
                    {   // Out: 144000
                        Convolution<float>({38,38,512}, {{3,3,512}, 4*(21+4), 1,PaddingType::SAME}, RELU, CPU),Flatten<float>({38,38,4*(21+4)},CPU)
                    },
                    {
                        MaxPool<float>({38,38,512},{{1,1},2},nullptr,CPU),

                        //conv_5
                        Convolution<float>({19,19,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),
                        Convolution<float>({19,19,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU),                
                        Convolution<float>({19,19,512}, {{3,3,512}, 512, 1,PaddingType::SAME}, RELU, CPU), 

                        //conv_6      
                        Convolution<float>({19,19,512}, {{3,3,512}, 1024, 1,PaddingType::SAME}, RELU, CPU),
                        
                        //conv_7
                        Convolution<float>({19,19,512}, {{1,1,1024}, 1024, 1,PaddingType::SAME}, RELU, CPU),
                        Concat<float>({19,19,1024},{73900,1,1},{ // Out: 73900
                            {   // Out: 54150
                                Convolution<float>({19,19,1024}, {{3,3,1024}, 6*(21+4), 1,PaddingType::SAME}, RELU, CPU),Flatten<float>({19,19,6*(21+4)},CPU)
                            },
                            {
                                // conv_8_2
                                Convolution<float>({19,19,1024}, {{1,1,1024}, 256, 1,PaddingType::SAME}, RELU, CPU),
                                MaxPool<float>({19,19,256},{{1,1},2},nullptr,CPU),
                                Convolution<float>({10,10,256}, {{1,1,256}, 512, 1,PaddingType::SAME}, RELU, CPU),
                                Concat<float>({10,10,512},{19750,1,1},{ // Out: 19750
                                    {  // Out: 15000
                                    Convolution<float>({10,10,512}, {{3,3,512}, 6*(21+4), 1,PaddingType::SAME}, RELU, CPU),Flatten<float>({10,10,6*(21+4)},CPU) 
                                    },
                                    {
                                        // conv_9_2
                                        Convolution<float>({10,10,512}, {{1,1,1024}, 128, 1,PaddingType::SAME}, RELU, CPU),
                                        MaxPool<float>({10,10,128},{{1,1},2},nullptr,CPU),
                                        Convolution<float>({5,5,128}, {{1,1,128}, 256, 1,PaddingType::SAME}, RELU, CPU),
                                        Concat<float>({5,5,256},{4750,1,1},{ // Out: 4750
                                            {   // Out: 3750
                                                Convolution<float>({5,5,256}, {{3,3,256}, 6*(21+4), 1,PaddingType::SAME}, RELU, CPU),Flatten<float>({5,5,6*(21+4)},CPU) 
                                            },
                                            { 
                                                Convolution<float>({5,5,256}, {{1,1,256}, 128, 1,PaddingType::SAME}, RELU, CPU),
                                                MaxPool<float>({5,5,128},{{1,1},2},nullptr,CPU),
                                                Convolution<float>({3,3,128}, {{1,1,128}, 256, 1,PaddingType::SAME}, RELU, CPU),
                                                Concat<float>({3,3,256},{1000,1,1},{ // Out 1000
                                                    {   // Out: 900
                                                        Convolution<float>({3,3,256}, {{3,3,256}, 4*(21+4), 1,PaddingType::SAME}, RELU, CPU),Flatten<float>({3,3,4*(21+4)},CPU) 
                                                    },
                                                    {
                                                        Convolution<float>({3,3,256}, {{1,1,256}, 128, 1,PaddingType::SAME}, RELU, CPU),
                                                        MaxPool<float>({3,3,128},{{2,2},2},nullptr,CPU),
                                                        Convolution<float>({1,1,128}, {{1,1,128}, 256, 1,PaddingType::SAME}, RELU, CPU),
                                                        Convolution<float>({1,1,256}, {{3,3,256}, 4*(21+4), 1,PaddingType::SAME}, RELU, CPU),
                                                        Flatten<float>({1,1,4*(21+4)},CPU) // Out: 100                        
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
