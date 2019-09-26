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
    int err;                            // error code returned from api calls
    float* input_h   = new float[INPUT_SIZE]();
    float *results_h = new float[IMG_SIZE*IMG_SIZE]();  // results returned from device
    cl_mem conv_kernels_d[16];
    float* conv_kernels_h[16];
    //                                   0    1   2   3   4   5    6    7    8    9   10   11  12  13  14  15
    const unsigned int conv_ch[16] = {IMG_CH, 8,  32, 32, 64,  64, 128, 128, 256, 64, 128, 32, 64, 32, 32, 16};
    const unsigned int  n_conv[16] = {8,      32, 32, 64, 64, 128, 128, 128,  64, 64, 32,  32, 32, 32, 16, 1};
    // const unsigned int conv_ch[16] = {IMG_CH, 8,  32, 32, 64, 64,  128, 128, 256, 128, 192, 64, 96, 32, 32, 16};
    // const unsigned int  n_conv[16] = {8,      32, 32, 64, 64, 128, 128, 128, 128, 128, 64,  64, 32, 32, 16, 1};
    const unsigned int conv_sz[16] = {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1};

    std::ifstream weights_fin("tomogan_weights_serilize.bin", std::ios::binary);
    unsigned int total_params = 0;
    for(int i = 0; i < 16; i++){
        unsigned int n_weights = (conv_sz[i] * conv_sz[i] * conv_ch[i] * n_conv[i]);
        unsigned int buf_size = sizeof(float) * n_weights;
        total_params += n_weights;
        printf("%6d paras (%6d Bytes) for conv2d_%02d kernel in_ch: %3d, no_ch: %3d\n", n_weights, buf_size, i, conv_ch[i], n_conv[i]);
        conv_kernels_h[i] = new float[buf_size]();
        weights_fin.read((char *) conv_kernels_h[i], buf_size);
        if(weights_fin){
            continue;
            // printf("%ld bytes of weights for conv %02d have been successfully read\n", weights_fin.gcount(), i);
        }else{
            printf("Error while load weights for conv %02d, EoF reached, only %ld bytes could be read\n", i, weights_fin.gcount());
            exit(-1);
        }
    }
    weights_fin.close();

    std::ifstream inputs_fin("test_input_serilize.bin", std::ios::binary);
    inputs_fin.read((char *) input_h, sizeof(float) * INPUT_SIZE);
    if(inputs_fin){
        printf("%ld bytes of input data have been successfully read\n", inputs_fin.gcount());
    }else{
        printf("Error while load input, EoF reached, only %ld bytes could be read\n", inputs_fin.gcount());
        exit(-1);
    }
    inputs_fin.close();


    auto comp_st = chrono::steady_clock::now();
    auto comp_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to compute on device!\n", \
           chrono::duration_cast<chrono::microseconds>(comp_ed - comp_st).count()/1000.);

    // dump output array to a file
    std::ofstream img_fout("output_img.bin", std::ios::out | std::ios::binary);
    img_fout.write((char *) results_h, sizeof(float) * OUTPUT_SIZE);
    img_fout.close();

    delete[] results_h;
    delete[] input_h;
}