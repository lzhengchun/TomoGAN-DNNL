#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <chrono>

#include "main.hpp"

#ifdef __APPLE__
    #define CL_SILENCE_DEPRECATION 
    #include <OpenCL/opencl.h>
#else
    #define CL_TARGET_OPENCL_VERSION 210
    #include <CL/cl.h>
#endif

using namespace std;

#define IMG_SIZE    (1024)
#define IMG_WIDTH   IMG_SIZE
#define IMG_HEIGHT  IMG_SIZE
#define IMG_CH      (3)
#define INPUT_SIZE  (IMG_SIZE * IMG_SIZE * IMG_CH)
#define OUTPUT_SIZE (IMG_SIZE * IMG_SIZE)
#define BOX1_IMG_SIZE (IMG_SIZE)
#define BOX2_IMG_SIZE (IMG_SIZE/2)
#define BOX3_IMG_SIZE (IMG_SIZE/4)
#define INTR_IMG_SIZE (IMG_SIZE/8)

#define MAX_SOURCE_SIZE (0x100000)

int main(int argc, char** argv)
{
    auto comp_st = chrono::steady_clock::now();

    tomogan(dnnl::engine::kind::gpu, 1024);

    auto comp_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to build model, initialize and compute on device!\n", \
           chrono::duration_cast<chrono::microseconds>(comp_ed - comp_st).count()/1000.);
}