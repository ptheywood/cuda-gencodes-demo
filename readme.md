# CUDA Gencode Demo

A Simple CUDA program to demonstrate how using different CUDA gencode arguments and different CUDA versions will result in different errors on different CUDA devices. 

## Gencodes

CUDA's [two-stage compilation process](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architectures) requires a *virtual architecture* and a *real architecture* to target the CUDA architecture to compile for. 

There are many ways to specify these flags to nvcc, but for simplicity this will just use the combined `-gencode` flag.

`-gencode=arch=compute_80,code=sm_80` will optimise for the real SM 80 (GA100) architecture, producing binaries which can be executed on any Virtual SM80 compatible device (any Ampere or Ada Lovelace GPU).

To support future architectures (i.e. Hopper, SM90+) then either a second explicit target needs adding, or PTX can be embedded by specifying the virtual architecture as both the `arch` and `code`: `-gencode=arch=compute_80,code=compute_80`.

```console
$ nvcc main.cu -o main -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80
$ ./main
Compiled with nvcc 12.1.66
__CUDA_ARCH_LIST__ 700
GPU 0: sm_86 NVIDIA GeForce RTX 3060 Ti
Hello from thread 0
```

If PTX is not included, and a non-compatible real architecture is specified, an error will occur on the first kernel launch (or some other cuda function which may trigger a similar error):

```console
$ nvcc main.cu -o main -gencode=arch=compute_70,code=sm_70
$ ./main 
Compiled with nvcc 12.1.66
__CUDA_ARCH_LIST__ 700
GPU 0: sm_86 NVIDIA GeForce RTX 3060 Ti
Error reported by cudaGetLastError() after helloWorld launch.
  cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device
```

But if PTX is embedded, it will be fine (as long as the PTX is a lower compute capability than the device)

```console
$ nvcc main.cu -o main -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70
$ ./main 
Compiled with nvcc 12.1.66
__CUDA_ARCH_LIST__ 700
GPU 0: sm_86 NVIDIA GeForce RTX 3060 Ti
Hello from thread 0
```

## Ampere and CUDA 10.x

Ampere's support was added to CUDA in [CUDA 11.0](https://docs.nvidia.com/cuda/archive/11.0_GA/cuda-toolkit-release-notes/index.html#cuda-general-new-features), for the `SM_80` (GA100) architecture.

SM_86 support was added in [CUDA 11.1](https://docs.nvidia.com/cuda/archive/11.1.0/cuda-toolkit-release-notes/index.html#cuda-general-new-features).

This means that a CUDA 10.x (or older) CUDA version cannot target Ampere, so PTX must be embedded for an older architecture

I.e. Compiling with sm_80 as a target will fail when using CUDA 10.x

```console
$ nvcc main.cu -o main -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80
nvcc fatal   : Unsupported gpu architecture 'compute_80'
```

Or compiling for SM86 will fail with CUDA 11.0:

```console
$ nvcc main.cu -o main -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86
nvcc fatal   : Unsupported gpu architecture 'compute_86'
```

Instead, PTX of one of the arch's supported by the CUDA version should be used, ideally the latest possible, though a more recent CUDA would be able to optimise for the newer GPUs better.

E.g. when compiling with CUDA 10.2 and ideally running on an SM_86 device, either specify an SM 7x virtual arch (and optionally a real sm7x arch, though this is not neccessary)

```console
$ nvcc main.cu -o main -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70
$ ./main 
Compiled with nvcc 10.2.89
GPU 0: sm_86 NVIDIA GeForce RTX 3090
Hello from thread 0
```

```console
$ nvcc main.cu -o main -gencode=arch=compute_70,code=compute_70
$ ./main 
Compiled with nvcc 10.2.89
GPU 0: sm_86 NVIDIA GeForce RTX 3090
Hello from thread 0
```
