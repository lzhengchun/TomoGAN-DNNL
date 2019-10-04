#ifndef TOMOGAN_HPP
#define TOMOGAN_HPP

#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <string>

#define CL_TARGET_OPENCL_VERSION 220
#include "dnnl.hpp"

#define DNNL_ARG_WEIGHTS_CUS 5671

// Read from memory, write to handle
inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

#if DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
#ifdef DNNL_USE_SYCL_BUFFERS
        auto buffer = mem.get_sycl_buffer<uint8_t>();
        auto src = buffer.get_access<cl::sycl::access::mode::read>();
        uint8_t *src_ptr = src.get_pointer();
#elif defined(DNNL_USE_DPCPP_USM)
        uint8_t *src_ptr = (uint8_t *)mem.get_data_handle();
#else
#error "Not expected"
#endif
        std::copy(src_ptr, src_ptr + size, (uint8_t *)handle);
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        dnnl::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();

        cl_int ret = clEnqueueReadBuffer(
                q, m, CL_TRUE, 0, size, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error("clEnqueueReadBuffer failed. Status Code: "
                    + std::to_string(ret) + "\n");
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        std::copy(src, src + size, (uint8_t *)handle);
        return;
    }

    assert(!"not expected");
}

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

#if DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
#ifdef DNNL_USE_SYCL_BUFFERS
        auto buffer = mem.get_sycl_buffer<uint8_t>();
        auto dst = buffer.get_access<cl::sycl::access::mode::write>();
        uint8_t *dst_ptr = dst.get_pointer();
#elif defined(DNNL_USE_DPCPP_USM)
        uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
#else
#error "Not expected"
#endif
        std::copy((uint8_t *)handle, (uint8_t *)handle + size, dst_ptr);
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        dnnl::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();

        cl_int ret = clEnqueueWriteBuffer(
                q, m, CL_TRUE, 0, size, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error(
                    "clEnqueueWriteBuffer failed. Status Code: "
                    + std::to_string(ret) + "\n");
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        std::copy((uint8_t *)handle, (uint8_t *)handle + size, dst);
        return;
    }

    assert(!"not expected");
}

#endif