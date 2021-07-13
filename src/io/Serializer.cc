#include <emptyNN/io/Serializer.hpp>

namespace emptyNN
{
    namespace io
    {
        template <class Type>
        Serializer<Type>::Serializer(std::string filename): filename(filename) {}

        

        template <class Type>
        void Serializer<Type>::dumpBinaryWeights(Sequential<Type>* model) {
            std::ofstream of(this->filename, std::ios::binary | std::ios::out);
            for(auto l: model->layers) {
               *l << of;
            }

        }

        template <class Type>
        void Serializer<Type>::loadBinaryWeights(Sequential<Type>* model) {
            std::ifstream ifs(this->filename, std::ios::binary | std::ios::out);
            for(auto l: model->layers) {
               //*l >> ifs;
            }
        }        
        REGISTER_CLASS(Serializer,float)
    } // namespace io
    
} // namespace emptyNN
