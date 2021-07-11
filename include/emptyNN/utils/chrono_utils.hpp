#include <chrono>
#include <functional>
using  s = std::chrono::seconds;
using  ns = std::chrono::nanoseconds;
using  ms = std::chrono::milliseconds;
using get_time = std::chrono::steady_clock;

void chronoIt(std::function<void(void)> codeBlock) {
    auto start = get_time::now();
    codeBlock();
    auto end = get_time::now();
    std::cout << "Frame-time: " << double(std::chrono::duration_cast<ns>(end - start).count())/1e6 << " ms";
    std::cout << " (FPS: " << 1/(double(std::chrono::duration_cast<ns>(end - start).count())/1e9) << " )\n";  
}

void chronoIt(std::function<void(void)> codeBlock, std::function<void(double)> callback) {
    auto start = get_time::now();
    codeBlock();
    auto end = get_time::now();
    callback(double(std::chrono::duration_cast<ns>(end - start).count()));
}