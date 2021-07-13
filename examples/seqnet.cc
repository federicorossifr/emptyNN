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
    seq = Models::VGG16<floatx>();
    io::Serializer<floatx> ser("test_fp.bin");
    seq->summary();
    ser.dumpBinaryWeights(seq);
    std::cout << "Loaded model" << std::endl;
}

