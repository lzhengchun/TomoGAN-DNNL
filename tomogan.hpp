#ifndef TOMOGAN_HPP
#define TOMOGAN_HPP

#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <string>

#define CL_TARGET_OPENCL_VERSION 220
#include "dnnl.hpp"

#define DNNL_ARG_WEIGHTS_CUS 5671

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

    // If all above fails, the example should be ran properly
    std::cerr << "Please run example like this" << argv[0] << " cpu|gpu";
    if (extra_args) { std::cerr << " [extra arguments]"; }
    std::cerr << "\n";
    exit(1);
}

// Read from memory, write to handle
inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        std::copy(src, src + bytes, (uint8_t *)handle);
    }
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    else if (eng.get_kind() == dnnl::engine::kind::gpu) {
        dnnl::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();

        cl_int ret = clEnqueueReadBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error("clEnqueueReadBuffer failed. Status Code: "
                    + std::to_string(ret) + "\n");
    }
#endif
}

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        std::copy((uint8_t *)handle, (uint8_t *)handle + bytes, dst);
    }
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    else if (eng.get_kind() == dnnl::engine::kind::gpu) {
        dnnl::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();
        size_t bytes = mem.get_desc().get_size();

        cl_int ret = clEnqueueWriteBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error(
                    "clEnqueueWriteBuffer failed. Status Code: "
                    + std::to_string(ret) + "\n");
    }
#endif
}

#endif