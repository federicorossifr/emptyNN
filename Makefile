CC=$(CXX)


OPTIONS := -fPIC -DUSE_POSIT
ARCH_FLAGS := -march=native
EXTRA_C_FLAGS := -Ofast ${ARCH_FLAGS}  -std=c++17 -fopenmp ${OPTIONS}
# EXTRA_C_FLAGS := -g -std=c++17 


include Makefile.objs
