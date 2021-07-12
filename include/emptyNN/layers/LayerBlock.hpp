#pragma once
#include <emptyNN/Sequential.hpp>
#include <emptyNN/Layer.hpp>
#include <vector>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class LayerBlock: public Layer<Type> {
            protected:
                std::vector<std::vector<Layer<Type>*>> block;
            public:
                LayerBlock(Shape in, Shape out);   
                LayerBlock(Shape in, Shape out,std::vector<std::vector<Layer<Type>*>> block);   
                virtual void forward();
                virtual ~LayerBlock();
                void summary();
                virtual Type* merge(Type* tensors[]) = 0;
                virtual std::ostream& operator<<(std::ostream& out) {return out;}
        };
    } // namespace Layers
} // namespace emptyNN