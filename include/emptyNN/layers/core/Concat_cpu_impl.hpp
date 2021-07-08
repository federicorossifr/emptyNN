#pragma once
#include <emptyNN/layers/LayerBlock.hpp>

namespace emptyNN {
    namespace Layers {
        namespace Impl
        {
            template <class Type>
            class ConcatCPUImpl: public LayerBlock<Type> {

                public:
                    ConcatCPUImpl(Shape in, Shape out,std::vector<std::vector<Layer<Type>*>> _block);      
                    virtual Type* merge(Type* tensors[]);
                    virtual void activate() {};
                    virtual void backward() {};
            };

        } // namespace Impl    
    } // namespace Layers    
} // namespace emptyNN