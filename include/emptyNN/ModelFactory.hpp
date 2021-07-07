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
    } // namespace Models
    
} // namespace emptyNN
