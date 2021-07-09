#pragma once
#include <emptyNN/Layer.hpp>
#include <cassert>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class Flatten: public Layer<Type> {;
            public:
                Flatten(Shape in);
                virtual void forward();
                virtual void backward() {};
                virtual void activate() {};
        };        
    } // namespace Layers
} // namespace emptyNN
