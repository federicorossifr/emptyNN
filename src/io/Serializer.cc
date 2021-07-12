#include <emptyNN/io/Serializer.hpp>


namespace emptyNN
{
    namespace io
    {
        template <class Type>
        Serializer<Type>::Serializer(std::string filename): filename(filename) {}

        template <class Base, class T>
        inline bool instanceof(const T*) {
            return std::is_base_of<Base, T>::value;
        }
        REGISTER_CLASS(Serializer,float)
    } // namespace io
    
} // namespace emptyNN
