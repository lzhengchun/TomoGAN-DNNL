#include <iostream>

#ifdef __APPLE__
    #define CL_SILENCE_DEPRECATION 
    #include <OpenCL/opencl.h>
#else
    #define CL_TARGET_OPENCL_VERSION 210
    #include <CL/cl.h>
#endif
#include "dnnl.hpp"

using namespace std;
using namespace dnnl;

#define oclErrchk(ans)  OCLAssert((ans), __FILE__, __LINE__) 
inline void OCLAssert(int code, string file, int line){
    if (code != CL_SUCCESS){
        cerr << "OpenCL Error: " << code << "; file: " << file << ", line:" << line << endl;
        exit(-1);
    }
}

static dnnl::engine::kind parse_engine_kind(
        int argc, char **argv, int extra_args = 0) {
    // Returns default engine kind, i.e. CPU, if none given
    if (argc == 1) {
        return dnnl::engine::kind::cpu;
    } else if (argc <= extra_args + 2) {
        std::string engine_kind_str = argv[1];
        // Checking the engine type, i.e. CPU or GPU
        if (engine_kind_str == "cpu") {
            return dnnl::engine::kind::cpu;
        } else if (engine_kind_str == "gpu") {
            // Checking if a GPU exists on the machine
            if (dnnl::engine::get_count(dnnl::engine::kind::gpu) == 0) {
                std::cerr << "Application couldn't find GPU, please run with "
                             "CPU instead. Thanks!\n";
                exit(1);
            }
            return dnnl::engine::kind::gpu;
        }
    }

    // If all above fails, the example should not be ran properly
    std::cerr << "Please run example like this" << argv[0] << " cpu|gpu";
    if (extra_args) { std::cerr << " [extra arguments]"; }
    std::cerr << "\n";
    exit(1);
}

extern void tomogan(dnnl::engine::kind engine_kind, unsigned int img_sz);