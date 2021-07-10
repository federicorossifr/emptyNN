#include "emptyNN/activations/Elu.hpp"
namespace emptyNN {
    namespace Activations {
        template <class Type>
        EluFunctor<Type>::EluFunctor(Type alpha): alpha(alpha){};
        
        template <class Type>
        void EluFunctor<Type>::operator()(Type* in_tensor, Shape in_shape) { 
            #pragma omp parallel for simd
            for(size_t i = 0; i < in_shape.size(); ++i) {
                Type& el = in_tensor[i];
                el = (el >= Type(0.f))? el : alpha*(std::exp(el)-Type(1.f)); 
            }
        }

        template <class Type>
        Type EluFunctor<Type>::grad(Type el) { return (el >= Type(0.f))? Type(1.f) : alpha*( std::exp(el) ); };

        
        REGISTER_CLASS(EluFunctor,float);
    }
}