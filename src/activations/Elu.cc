#include "emptyNN/activations/Elu.hpp"
namespace emptyNN {
    namespace Activations {
        template <class Type>
        EluFunctor<Type>::EluFunctor(Type alpha): alpha(alpha){};
        
        template <class Type>
        Type EluFunctor<Type>::operator()(Type el) { return (el >= Type(0.f))? el : alpha*(std::exp(el)-Type(1.f)); }

        template <class Type>
        Type EluFunctor<Type>::grad(Type el) { return (el >= Type(0.f))? Type(1.f) : alpha*( std::exp(el) ); };

        
        REGISTER_CLASS(EluFunctor,float);
    }
}