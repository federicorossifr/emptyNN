#pragma once
#include <emptyNN/Layer.hpp>
#include <cassert>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        class Dense: public Layer<Type> {
            protected:
                Type* connections;
                Type* bias;
            public:
                Dense(Shape in, Shape out,Activation<Type>* a);
                virtual ~Dense();
                virtual void summary() {
                    Layer<Type>::summary(); 
                    std::cout << "Dense weights: " << this->i_shape.size() * this->o_shape.size() + this->o_shape.size() << std::endl; 
                }

                virtual std::ostream& operator<<(std::ostream& out);
                virtual std::istream& operator>>(std::istream& ifs);                

        };        
    } // namespace Layers
} // namespace emptyNN
