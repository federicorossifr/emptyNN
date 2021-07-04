CC=$(CXX)
EXTRA_C_FLAGS := -O3
LAYER_OBJS := Layer.o Conv.o Dense.o
ACT_OBJS := Activation.o Elu.o
IMPL_OBJS := ConvCpuImpl.o
MODEL_OBJS := Model.o Sequential.o
INCL_FLAGS := -I./include/

Activation.o: src/Activation.cc include/emptyNN/Activation.hpp
	$(CC) -c ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} src/Activation.cc -o Activation.o



Elu.o: src/activations/Elu.cc include/emptyNN/activations/Elu.hpp include/emptyNN/Activation.hpp
	$(CC) -c ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} src/activations/Elu.cc -o Elu.o

Layer.o: src/Layer.cc include/emptyNN/Layer.hpp
	$(CC) -c ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} src/Layer.cc -o Layer.o

Conv.o: src/layers/Conv.cc include/emptyNN/layers/Conv.hpp
	$(CC) -c ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} src/layers/Conv.cc -o Conv.o

Dense.o: src/layers/Dense.cc include/emptyNN/layers/Dense.hpp
	$(CC) -c ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} src/layers/Dense.cc -o Dense.o

ConvCpuImpl.o: src/layers/core/Conv_cpu_impl.cc include/emptyNN/layers/core/Conv_cpu_impl.hpp
	$(CC) -c ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} src/layers/core/Conv_cpu_impl.cc -o ConvCpuImpl.o

Layers: ${LAYER_OBJS}

Activations: ${ACT_OBJS}

LayersImpl: ${IMPL_OBJS}

Models: ${MODEL_OBJS}

Types.o: src/Types.cc
	$(CC) -c ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} src/Types.cc -o Types.o

Factory.o: include/emptyNN/Factory.hpp src/Factory.cc
	$(CC) -c ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} src/Factory.cc -o Factory.o

Model.o: include/emptyNN/Model.hpp src/Model.cc
	$(CC) -c ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} src/Model.cc -o Model.o

Sequential.o: include/emptyNN/Sequential.hpp src/Sequential.cc
	$(CC) -c ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} src/Sequential.cc -o Sequential.o


seqnet: examples/seqnet.cc Layers LayersImpl Models Factory.o Types.o Activations 
	$(CC) ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} ${LAYER_OBJS} ${ACT_OBJS} ${IMPL_OBJS} ${MODEL_OBJS} Types.o Factory.o examples/seqnet.cc -o seqnet.exe

examples: seqnet

clean:
	rm -rf *.o

