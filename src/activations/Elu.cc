#include "emptyNN/activations/Elu.hpp"
namespace emptyNN {
    namespace Activations {
        template <class Type>
        EluFunctor<Type>::EluFunctor(Type alpha): alpha(alpha){};
        
        template <class Type>
        Type EluFunctor<Type>::operator()(Type el) { return (el >= 0)? el : alpha*(std::exp(el)-1); }

        template <class Type>
        Type EluFunctor<Type>::grad(Type el) { return (el >= 0)? 1 : alpha*( std::exp(el) ); };

        
        template class EluFunctor<float>;
    }
}