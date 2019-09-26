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

extern void tomogan(dnnl::engine::kind engine_kind, unsigned int img_sz);