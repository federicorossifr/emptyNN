#pragma once
#include "emptyNN/Types.hpp"
#include "emptyNN/Activation.hpp"
namespace emptyNN {

    namespace io
    {
        template <class Type>
        class Serializer;        
    } // namespace io

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
            virtual ~Layer();
            void fillInTensor(Type* i);
            Shape getOutputShape();
            Shape getInputShape();
            virtual void summary();
            
            Type* operator()();
            virtual void forward() = 0;
            virtual void backward(Type* grad) = 0;
            virtual void activate();

            virtual std::ostream& operator<<(std::ostream& out) = 0;
            virtual std::istream& operator>>(std::istream& ifs) = 0;

            friend class io::Serializer<Type>;
    };
}