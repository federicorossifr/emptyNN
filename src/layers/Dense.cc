#include "emptyNN/layers/Dense.hpp"
#include <emptyNN/utils/tensor_utils.hpp>

#include <iostream>
namespace emptyNN {
    namespace Layers {
        template <class Type>
        Dense<Type>::Dense(Shape in, Shape out,Activation<Type>* a): Layer<Type>(in,out,a) {
            // assert(in.depth == 1 && out.depth == 1);
            // In-vector   1 x (in.height * in.width) 
            // Out-vector  (out.height * out.width) x 1
            // Connection matrix (in.height * in.width) x (out.height * out.width)
            size_t conn_size = in.size() * out.size();
            connections = new Type[conn_size];
            bias = new Type[out.size()];
            emptyNN::Utils::Tensors::fillRandomUniform<Type>(connections,conn_size);
            emptyNN::Utils::Tensors::fillRandomUniform<Type>(bias,out.size());

        }

        template <class Type>
        Dense<Type>::~Dense() {
            delete[] connections;
            delete[] bias;
        }    

        template <class Type>
        std::ostream& Dense<Type>::operator<<(std::ostream& ofs) {
            Shape in = this->i_shape;
            Shape out = this->o_shape;
            size_t connection_size = in.size() * out.size();
            ofs.write(reinterpret_cast<char*>(connections),connection_size*sizeof(Type))
               .write(reinterpret_cast<char*>(bias),out.size()*sizeof(Type));
            return ofs;
        }        

        template <class Type>
        std::istream& Dense<Type>::operator>>(std::istream& ifs) {
            Shape in = this->i_shape;
            Shape out = this->o_shape;
            size_t connection_size = in.size() * out.size();
            ifs.read(reinterpret_cast<char*>(connections),connection_size*sizeof(Type))
                .read(reinterpret_cast<char*>(bias),out.size()*sizeof(Type));
            return ifs;
        }              

        REGISTER_CLASS(Dense,float);

    }
}