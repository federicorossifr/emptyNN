#pragma once
#include <emptyNN/Layer.hpp>
#include <cassert>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class Dense: public Layer<Type> {
            protected:
                Type* connections;
            public:
                Dense(Shape in, Shape out);
                virtual ~Dense();
        };        
    } // namespace Layers
} // namespace emptyNN
