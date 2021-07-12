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
                virtual std::ostream& operator<<(std::ostream& out)  {return out;}
        };        
    } // namespace Layers
} // namespace emptyNN
