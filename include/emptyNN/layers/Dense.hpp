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
                Dense(Shape in, Shape out,Activation<Type>* a);
                virtual ~Dense();
                virtual std::ostream& operator<<(std::ostream& out);
        };        
    } // namespace Layers
} // namespace emptyNN
