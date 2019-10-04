#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <chrono>

#include "main.hpp"

using namespace std;

int main(int argc, char** argv)
{
    try{
            auto comp_st = chrono::steady_clock::now();
            tomogan(parse_engine_kind(argc, argv), 1024);
            auto comp_ed = chrono::steady_clock::now();
            printf("It takes %.3f ms to build model, initialize and compute on device!\n", \
                chrono::duration_cast<chrono::microseconds>(comp_ed - comp_st).count()/1000.);
    }catch(error &e){
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
