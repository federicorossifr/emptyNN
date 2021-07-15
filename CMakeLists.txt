cmake_minimum_required(VERSION 3.10)

project(emptyNN)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(ARCH_FLAGS "-march=native -Ofast")

option(USE_ANYFLOAT "Enable support for cppPosit posit and anyfloat emulation" OFF)
option(USE_OPENMP   "Enable multithread support for DNN kernels"               ON)
option(USE_RVV      "Enable RISC-V Vector support"                             OFF)

if(USE_RVV)
    message("Using RVV-Compiler optimizations")
    set(ARCH_FLAGS "-mepi -O3 -DUSE_RVV")
endif(USE_RVV)

if(USE_OPENMP)
    message("Using OPENMP multithreaded kernels")
    set(ARCH_FLAGS "${ARCH_FLAGS} -fopenmp")
endif(USE_OPENMP)

if(USE_ANYFLOAT)
    message("Adding include directories for cppPosit")
    set(ANY_FLOAT_INCL_DIR $ENV{CPPPOSIT_INCLUDE_DIR})
    include_directories(${ANY_FLOAT_INCL_DIR})
    set(ARCH_FLAGS "${ARCH_FLAGS} -DUSE_POSIT")    
endif(USE_ANYFLOAT)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
message("CXX FLAGS:${CXX} ${CMAKE_CXX_FLAGS}")


file(GLOB_RECURSE emptynn_sources "src/**/*.cc")
file(GLOB_RECURSE emptynn_sources "src/*.cc")
include_directories("${PROJECT_SOURCE_DIR}/include")



add_library(emptynn SHARED ${emptynn_sources})



add_custom_target(examples)

add_executable(resnet.exe "examples/resnet.cc")
target_link_libraries(resnet.exe PUBLIC emptynn)
add_dependencies(examples resnet.exe)



add_executable(seqnet.exe "examples/seqnet.cc")
target_link_libraries(seqnet.exe PUBLIC emptynn)
add_dependencies(examples seqnet.exe)


if(USE_ANYFLOAT)
    add_executable(resnet_p160.exe "examples/resnet.cc")
    target_link_libraries(resnet_p160.exe PUBLIC emptynn)
    target_compile_definitions(resnet_p160.exe PUBLIC TYPE=Posit16_0)
    add_dependencies(examples resnet_p160.exe)
endif(USE_ANYFLOAT)

