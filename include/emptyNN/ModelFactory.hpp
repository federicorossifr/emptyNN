#include <emptyNN/Factory.hpp>
#include <emptyNN/Sequential.hpp>

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
                #define RELU Factory::Activations::Elu<Type>(1.)
                #define SMAX Factory::Activations::Elu<Type>(1.)
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
                sc =Factory::Layers::MaxPool<Type>({tmp.width,tmp.height,256},{{1,1},2},nullptr,CPU),

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
    } // namespace Models
    
} // namespace emptyNN
