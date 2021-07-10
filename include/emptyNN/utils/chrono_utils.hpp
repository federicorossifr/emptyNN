#include <chrono>
using  s = std::chrono::seconds;
using  ns = std::chrono::nanoseconds;
using  ms = std::chrono::milliseconds;
using get_time = std::chrono::steady_clock ;

template <class Fun>
void chronoIt(Fun codeBlock) {
    auto start = get_time::now();
    codeBlock();
    auto end = get_time::now();
    std::cout << "\rFPS: " << 1e9/double(std::chrono::duration_cast<ns>(end - start).count()) << std::endl;    
}

template <class Fun, class Fun2>
void chronoIt(Fun codeBlock, Fun2 callback) {
    auto start = get_time::now();
    codeBlock();
    auto end = get_time::now();
    callback(double(std::chrono::duration_cast<ns>(end - start).count()));
}