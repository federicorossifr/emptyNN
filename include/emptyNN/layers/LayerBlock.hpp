#pragma once
#include <emptyNN/Sequential.hpp>
#include <emptyNN/Layer.hpp>

namespace emptyNN {
    namespace Layers {
        template <class Type>
        class LayerBlock: public Layer<Type> {
            protected:
                std::vector<std::vector<Layer<Type>*>> block;
            public:
                LayerBlock(Shape in, Shape out);      
                virtual void forward();
                virtual ~LayerBlock();
                virtual Type* merge(Type* tensors[]) = 0;
        };
    } // namespace Layers
} // namespace emptyNN