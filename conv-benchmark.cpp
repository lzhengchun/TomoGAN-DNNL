#include <assert.h>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <chrono>

#include "tomogan.hpp"
#include "dnnl.hpp"

using namespace std;
using namespace dnnl;
memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
            std::multiplies<memory::dim>());
}

int main(int argc, char const *argv[]){
    using tag = memory::format_tag;
    using dt  = memory::data_type;
    unsigned int img_sz = 1024;
    const memory::dim batch = 1;
    dnnl::engine eng(dnnl::engine::kind::gpu, 0);
    stream s(eng);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    //[Create network]

    // {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
    // strides: {4, 4}
    const memory::dim n_filter = 32;
    const memory::dim in_ch = 128;
    memory::dims conv1_src_tz = {batch, in_ch, img_sz, img_sz};
    memory::dims conv1_weights_tz = {n_filter, in_ch, 3, 3};
    memory::dims conv1_bias_tz = {n_filter};
    memory::dims conv1_dst_tz = {batch, n_filter, img_sz, img_sz};
    memory::dims conv1_strides = {1, 1};
    memory::dims conv1_padding = {1, 1};

    /// Allocate buffers for input and output data, weights, and bias.
    /// @snippet cnn_inference_f32.cpp Allocate buffers
    //[Allocate buffers]
    std::vector<float> user_src(batch * in_ch * img_sz * img_sz);
    for(auto n = 0; n < batch; n++)
        for(auto c = 0; c < in_ch; c++)
            for(auto h = 0; h < img_sz; h++)
                for(auto w = 0; w < img_sz; w++){
                    user_src[n * in_ch * img_sz * img_sz +\
                             c * img_sz * img_sz + \
                             h * img_sz + w] = c;
                }

    std::vector<float> conv1_weights(product(conv1_weights_tz));
    for(auto i = 0; i < conv1_weights.size(); i++){
        conv1_weights[i] = 1;
    }

    std::vector<float> conv1_bias(product(conv1_bias_tz));
    for(auto i = 0; i < conv1_bias.size(); i++){
        conv1_bias[i] = 0;
    }

    auto user_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_memory(user_src.data(), user_src_memory);

    auto user_weights_memory = memory({{conv1_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv1_weights.data(), user_weights_memory);

    auto conv1_user_bias_memory = memory({{conv1_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv1_bias.data(), conv1_user_bias_memory);
    //[Create user memory]

    auto conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);
    //[Create convolution memory descriptors]

    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
            conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding,
            conv1_padding);

    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);

    auto conv1_src_memory = user_src_memory;
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        std::cout << "!!!!! src mem needs reorder" << std::endl;
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                            {DNNL_ARG_TO, conv1_src_memory}});
    }

    auto conv1_weights_memory = user_weights_memory;
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        std::cout << "!!!!! weight mem needs reorder" << std::endl;
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        reorder(user_weights_memory, conv1_weights_memory)
                .execute(s, user_weights_memory, conv1_weights_memory);
    }

    auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);

    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_src_memory},
            {DNNL_ARG_WEIGHTS, conv1_weights_memory},
            {DNNL_ARG_BIAS, conv1_user_bias_memory},
            {DNNL_ARG_DST, conv1_dst_memory}});

    for (auto args : net_args){
        auto in_dims = args[DNNL_ARG_SRC].get_desc().data.dims;
        printf("Input: %d x %3d x %d x %d", in_dims[0], in_dims[1], in_dims[2], in_dims[3]);

        auto out_dims = args[DNNL_ARG_DST].get_desc().data.dims;
        printf(" => Output: %d x %3d x %d x %d\n", out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    }

    auto mdl_exe_st = chrono::steady_clock::now();
    auto rep_times = 1;
    for (int j = -3; j < rep_times; ++j) {
        if(j == 0){
            mdl_exe_st = chrono::steady_clock::now();
        }
        assert((net.size()) == net_args.size() && "something is missing\n");
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(s, net_args.at(i));
        s.wait();
    }
    
    auto mdl_exe_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to execute the model!\n", \
           chrono::duration_cast<chrono::microseconds>(mdl_exe_ed - mdl_exe_st).count()/1000./rep_times);
// check res

    // copy results from dnnl device to cpu mempry
    auto output_dims = net_args.back()[DNNL_ARG_DST].get_desc().data.dims;
    auto output_size = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
    auto output_buf = new float[output_size]();
    read_from_dnnl_memory(output_buf, net_args.back()[DNNL_ARG_DST]);
    double sum = 0;
    for(size_t i = 0; i < output_size; i++){
        sum += output_buf[i];
    }
    printf("results check sum: %lf\n", sum);
}
