#pragma once
#include "emptyNN/Types.hpp"
#include "emptyNN/Activation.hpp"
namespace emptyNN {
    template <class Type>
    class Layer {
        protected:
            Type* i_tensor;
            Shape i_shape;
            Type* o_tensor;
            Shape o_shape;
            Activation<Type>* activation;
        public:
            Layer(Shape in, Shape out,Activation<Type>* a = nullptr);
            Layer(Shape in,Activation<Type>* a = nullptr);
            void fillInTensor(Type* i);
            Type* operator()();
            virtual void forward() = 0;
            virtual void backward() = 0;
            virtual void activate() = 0;
    };
}