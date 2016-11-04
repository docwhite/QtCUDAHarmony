TARGET = app

HEADERS = \
    include/particle.h \
    include/ramon.h \
    include/v3.h

SOURCES = \
    src/main.cpp

# Those with __device__ __host__ decorators (will be compiled with nvcc)
CUDA_HYBRID_CPP_FILES = \
    src/v3.cpp \
    src/particle.cpp \
    src/ramon.cpp

# Pure .cu files need to be compiled with nvcc as well
CUDA_CU_FILES= \
    cu/kernel_advance_particle.cu \
    cu/kernel_add_second_to_first.cu

# Compile the pure c++ .cpp files into .o not linking the functions yet
CXXFLAGS = -c

# Directory where all the *_cuda.o and *.o objects will be dropped to
OBJECTS_DIR = build/obj

# Includes needed by both pure c++ files
INCLUDEPATH += $$PWD/include

# Include path needed by hybrid .cpp and pure .cu files
CUDA_INCLUDES = -I/usr/include/cuda -I$$PWD/include

# The building steps are as follows:
#   1. NVCC compiles hybrid sources to build/{name}_hybrid_cuda.o (will be having
#      bindings for both CPU and GPU code, all handled by nvcc).
#   2. NVCC compiles .cu files (usually will be kernels and the function
#      wrapping it so we can launch it from a pure c++ file) to build/{name}_cuda.o
#   3. G++ compiles and assembles .cpp pure c++ files into build/{name}.o,
#      unlinked (g++ -c flag)
#   4. NVCC grabs all the *_cuda.o and makes a dlink.o that will be used by the
#      g++ compiler to find where all the right definitions are
#   5. G++ grabs all the objects *_cuda.o *.o dlink.o and links them together
#      producing an executable

LIBS += $$OBJECTS_DIR/dlink.o
LIBS += -L/usr/lib64 -lcudart
LIBS += -L/usr/lib64/nvidia -lcuda

# STEP 1
cuda_hybrid.input = CUDA_HYBRID_CPP_FILES
cuda_hybrid.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_hybrid_cuda.o
cuda_hybrid.commands = /usr/bin/nvcc $$CUDA_INCLUDES -x cu -arch=sm_20 -dc -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda_hybrid.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda_hybrid

# STEP 2
cuda_kernels.input = CUDA_CU_FILES
cuda_kernels.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
cuda_kernels.commands = /usr/bin/nvcc $$CUDA_INCLUDES -arch=sm_20 -dc -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda_kernels.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda_kernels

# STEP 4
QMAKE_PRE_LINK = /usr/bin/nvcc $$CUDA_INCLUDES -arch=sm_20 -dlink $$OBJECTS_DIR/*_cuda.o -o $$OBJECTS_DIR/dlink.o

OTHER_FILES += README.md
