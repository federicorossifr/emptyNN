#include "emptyNN/ModelFactory.hpp"
#include "emptyNN/Sequential.hpp"
#include "emptyNN/io/BinarySerializer.hpp"
#include <iostream>
#include <new>
using namespace emptyNN;
using namespace Factory::Layers;
int main() {
    using floatx = Posit8_0;
    Random::globalSeed = 2021;
    Sequential<floatx>* seq;
    seq = Models::LeNet5<floatx>();
    io::Serializer<floatx> *ser =  new io::BinarySerializer<floatx>("test_P8.bin");
    ser->serialize(seq);
}

