#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <chrono>

#include "main.hpp"

using namespace std;

int main(int argc, char** argv)
{
    auto comp_st = chrono::steady_clock::now();

    for (size_t i = 0; i < 1; i++){
        tomogan(dnnl::engine::kind::gpu, 1024);
    }
    
    auto comp_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to build model, initialize and compute on device!\n", \
           chrono::duration_cast<chrono::microseconds>(comp_ed - comp_st).count()/1000.);
}