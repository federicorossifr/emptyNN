#pragma once
#include "emptyNN/Types.hpp"
#include "emptyNN/Activation.hpp"
namespace emptyNN {
    template <class Type>
    class Layer {
        protected:
            Tensor<Type> i_tensor;
            Shape i_shape;
            Tensor<Type> o_tensor;
            Shape o_shape;
            Activation<Type>* activation;
        public:
            Layer(Shape in, Shape out,Activation<Type>* a = nullptr);
            Layer(Shape in,Activation<Type>* a = nullptr);
            virtual ~Layer();
            void fillInTensor(Tensor<Type>&& in);
            Shape getOutputShape();
            Shape getInputShape();
            virtual void summary();
            
            Tensor<Type>& operator()();
            virtual void forward() = 0;
            virtual void backward() = 0;
            virtual void activate();
    };
}