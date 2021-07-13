#pragma once
#include <emptyNN/layers/LayerBlock.hpp>
namespace emptyNN {
    namespace Layers {
        namespace Impl
        {
            

        
            template <class Type>
            class ResidualBlockCPUImpl: public LayerBlock<Type> {
                ResBlockParams params;
                public:
                    ResidualBlockCPUImpl(Shape in, Shape out,ResBlockParams params);      
                    virtual Type* merge(Type* tensors[]);
                    virtual void activate() {};
                    virtual void backward(Type* grad) {};
            };

        } // namespace Impl    
    } // namespace Layers    
} // namespace emptyNN