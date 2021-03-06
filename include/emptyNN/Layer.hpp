/*
emptyNN
Copyright (C) 2021 Federico Rossi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
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
            virtual Type* backward(Type* grad) = 0;
            virtual void activate();

            virtual std::ostream& operator<<(std::ostream& out) = 0;
            virtual std::istream& operator>>(std::istream& ifs) = 0;

            friend class io::Serializer<Type>;
    };
}