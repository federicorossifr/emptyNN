#pragma once
#include <emptyNN/Layer.hpp>
#include <cassert>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class Pad: public Layer<Type> {
            public:
                Pad(Shape in, Shape pad);
                virtual void forward();
                virtual void backward() {};
                virtual void activate() {};
        };        
    } // namespace Layers
} // namespace emptyNN
