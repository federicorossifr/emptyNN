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
                virtual void backward(Type* grad) {};
                virtual void activate() {};
                virtual std::ostream& operator<<(std::ostream& out)  {return out;}
                virtual std::istream& operator>>(std::istream& ifs)  {return ifs;}           

        };        
    } // namespace Layers
} // namespace emptyNN
