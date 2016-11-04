# QtCUDAHarmony
This is an example for our group project to structure hybrid CUDA-C++ classes
and CUDA kernel .cu files at the same time and using a Qt project configuration
file that does all the right steps to compile everything.

## Compiling
Change directory to where the particlesCuda.pro lives and run qmake-qt4 to
generate a file out of the .pro file. ``qmake-qt4 particlesCuda.pro`` and then
simply ``make``

## Compilation / Linking Steps Explained
You can check out the .pro file and see all the steps that are carried on.

When making from the terminal have a look at the output. I will be explaining
each step after the code block.

```
(STEP 1)
/usr/bin/nvcc -I/usr/include/cuda -I/home/i7243466/Dropbox/programming/particlesCuda/include -x cu -arch=sm_20 -dc -o build/obj/v3_hybrid_cuda.o src/v3.cpp
/usr/bin/nvcc -I/usr/include/cuda -I/home/i7243466/Dropbox/programming/particlesCuda/include -x cu -arch=sm_20 -dc -o build/obj/particle_hybrid_cuda.o src/particle.cpp
/usr/bin/nvcc -I/usr/include/cuda -I/home/i7243466/Dropbox/programming/particlesCuda/include -x cu -arch=sm_20 -dc -o build/obj/ramon_hybrid_cuda.o src/ramon.cpp
(STEP 2)
/usr/bin/nvcc -I/usr/include/cuda -I/home/i7243466/Dropbox/programming/particlesCuda/include -arch=sm_20 -dc -o build/obj/kernel_advance_particle_cuda.o cu/kernel_advance_particle.cu
/usr/bin/nvcc -I/usr/include/cuda -I/home/i7243466/Dropbox/programming/particlesCuda/include -arch=sm_20 -dc -o build/obj/kernel_add_second_to_first_cuda.o cu/kernel_add_second_to_first.cu
(STEP 3)
g++ -c -pipe -O2 -g -pipe -Wall -Wp,-D_FORTIFY_SOURCE=2 -fstack-protector-strong --param=ssp-buffer-size=4 -grecord-gcc-switches -m64 -mtune=generic -O2 -Wall -W -D_REENTRANT -DQT_NO_DEBUG -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED -I/usr/lib64/qt4/mkspecs/linux-g++ -I. -I/usr/include/QtCore -I/usr/include/QtGui -I/usr/include -Iinclude -I. -o build/obj/main.o src/main.cpp
(STEP 4)
/usr/bin/nvcc -I/usr/include/cuda -I/home/i7243466/Dropbox/programming/particlesCuda/include -arch=sm_20 -dlink build/obj/*_cuda.o -o build/obj/dlink.o
(STEP 5)
g++ -Wl,-O1 -Wl,-z,relro -o app build/obj/v3_hybrid_cuda.o build/obj/particle_hybrid_cuda.o build/obj/ramon_hybrid_cuda.o build/obj/kernel_advance_particle_cuda.o build/obj/kernel_add_second_to_first_cuda.o build/obj/main.o    -L/usr/lib64 build/obj/dlink.o -L/usr/lib64 -lcudart -L/usr/lib64/nvidia -lcuda -lQtGui -lQtCore -lpthread 

```

### Step 1
**nvcc** compiles hybrid sources to `build/obj/{name}_hybrid_cuda.o`. These object
files will be having all the functions that are used but they won't be defined.
In other words they won't be linked, this is done later in step 4 & 5.

### Step 2
**nvcc** compiles the .cu files (which usually will be the kernels and the
functions wrapping them, so that we can call it from CPU pure c++ files by
predeclaring the function wrapper at the top). The files get dropped to
``build/obj/{name}_cuda.o``

### Step 3
**g++** compiles and assembles (no linking) pure .cpp c++ files into 
``build/obj/{name}.o``. The ``-c`` flag is passed because we do not want linking yet.


### Step 4
**nvcc** grabs all the ``*_cuda.o `` objects and generates a **dlink.o** file
that will help the **g++** linking step to find all the function definitions and
bind them. Google what ``nvcc -dlink``  does if you are uncertain about it.

### Step 5
**g++** collects all the ``*_cuda.o`` (objects with cuda code), ``*.o`` (pure
c++ objects) and the ``dlink.o`` (linking information for g++ to do the binding)
and links everything producing an executable.