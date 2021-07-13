#include "emptyNN/ModelFactory.hpp"
#include "emptyNN/Sequential.hpp"
#include "emptyNN/io/Serializer.hpp"
#include <iostream>
#include <new>
using namespace emptyNN;
using namespace Factory::Layers;
int main() {
    using floatx = float;
    Random::globalSeed = 2021;
    Sequential<floatx>* seq;
    seq = Models::LeNet5<floatx>();
    io::Serializer<floatx> ser("test_fp.bin");
    ser.dumpBinaryWeights(seq);
}

