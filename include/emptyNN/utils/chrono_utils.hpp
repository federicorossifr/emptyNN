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