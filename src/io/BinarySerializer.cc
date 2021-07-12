#include <emptyNN/io/BinarySerializer.hpp>


namespace emptyNN
{
    namespace io
    {
        template <class Type>
        BinarySerializer<Type>::BinarySerializer(std::string filename): Serializer<Type>(filename) {}

        

        template <class Type>
        void BinarySerializer<Type>::serialize(Sequential<Type>* model) {
            std::ofstream of(this->filename, std::ios::binary | std::ios::out);
            for(auto l: model->layers) {
               *l << of;
            }

        }

        template <class Type>
        Sequential<Type>* BinarySerializer<Type>::deserialize() {
            Sequential<Type>* seq = new Sequential<Type>("Test");
            return seq;
        }        
        REGISTER_CLASS(BinarySerializer,float)
    } // namespace io
    
} // namespace emptyNN
