CC=$(CXX)


OPTIONS := -fPIC -DUSE_POSIT
ARCH_FLAGS := -march=native
EXTRA_C_FLAGS := -Ofast ${ARCH_FLAGS}  -std=c++17 -fopenmp ${OPTIONS}
# EXTRA_C_FLAGS := -g -std=c++17 

include Makefile.objs


resnet_p8: examples/resnet.cc ${LIB_OBJS} 
	$(CC) -DTYPE=Posit8_0 ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} ${LIB_OBJS} examples/resnet.cc -o resnet_p8.exe	

resnet_p161: examples/resnet.cc ${LIB_OBJS} 
	$(CC) -DTYPE=Posit16_1 ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} ${LIB_OBJS} examples/resnet.cc -o resnet_p161.exe	

resnet_bf16: examples/resnet.cc ${LIB_OBJS}
	$(CC) -DTYPE=Bfloat16 ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} ${LIB_OBJS} examples/resnet.cc -o resnet_bf16.exe	

resnet_bf8: examples/resnet.cc ${LIB_OBJS}
	$(CC) -DTYPE=Bfloat8 ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} ${LIB_OBJS} examples/resnet.cc -o resnet_bf8.exe		


resnet_fp32: examples/resnet.cc ${LIB_OBJS}
	$(CC) -DTYPE=float ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} ${LIB_OBJS} examples/resnet.cc -o resnet_fp32.exe		

resnet_fp32_emu: examples/resnet.cc ${LIB_OBJS}
	$(CC) -DTYPE=FloatEmu ${EXTRA_C_FLAGS} $(CFLAGS) ${INCL_FLAGS} ${LIB_OBJS} examples/resnet.cc -o resnet_fp32_emu.exe		


resnets: resnet_p8 resnet_bf16 resnet_fp32 resnet_bf8 resnet_fp32_emu resnet_p161

