#include <assert.h>
#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>

#include "tomogan.hpp"

using namespace std;
using namespace dnnl;

memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
            std::multiplies<memory::dim>());
}

void conv2d(memory::dim batch, 
            memory::dim in_ch,
            memory::dim knl_sz,
            memory::dim n_knl,
            std::vector<primitive> &net,
            std::vector<std::unordered_map<int, memory>> &net_args,
            engine &eng, 
            stream &s){
    
    memory::dim in_img_h = net_args.back()[DNNL_ARG_DST].get_desc().data.dims[2];
    memory::dim in_img_w = net_args.back()[DNNL_ARG_DST].get_desc().data.dims[3];

    memory::dims conv_weights_tz = {n_knl, in_ch, knl_sz, knl_sz};
    memory::dims conv_bias_tz = {n_knl};
    memory::dims conv_dst_tz = {batch, n_knl, in_img_h, in_img_w};
    memory::dims conv_strides = {1, 1};

    memory::dims conv_padding = {knl_sz/2, knl_sz/2}; // padding as the same

    /// Create memory that describes data layout in the buffers. here we use
    /// tag::oihw for weights.
    auto user_weights_memory   = memory({{conv_weights_tz}, memory::data_type::f32, memory::format_tag::oihw}, eng);
    auto conv_user_bias_memory = memory({{conv_bias_tz},    memory::data_type::f32, memory::format_tag::x},    eng);
    
    /// Create memory descriptors with layout tag::any. The `any` format enables
    /// the convolution primitive to choose the data format that will result in
    /// best performance based on its input parameters (convolution kernel
    /// sizes, strides, padding, and so on). If the resulting format is different
    /// from `nchw`, the user data must be transformed to the format required for
    /// the convolution (as explained below).
    auto conv_bias_md    = memory::desc({conv_bias_tz},    memory::data_type::f32, memory::format_tag::any);
    auto conv_weights_md = memory::desc({conv_weights_tz}, memory::data_type::f32, memory::format_tag::any);
    auto conv_dst_md     = memory::desc({conv_dst_tz},     memory::data_type::f32, memory::format_tag::any);

    /// Create a convolution descriptor
    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, net_args.back()[DNNL_ARG_DST].get_desc(), conv_weights_md,
            conv_bias_md, conv_dst_md, conv_strides, conv_padding, conv_padding);

    /// Create a convolution primitive descriptor. Once created, this
    /// descriptor has specific formats instead of the `any` format specified
    /// in the convolution descriptor.
    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);
    
    /// Check whether data and weights formats required by convolution is different
    /// from the user format. In case it is different change the layout using reorder primitive.
    // auto user_src_memory = net_args.back()[DNNL_ARG_DST];
    // auto conv_src_memory = user_src_memory;
    // if (conv_prim_desc.src_desc() != user_src_memory.get_desc()) {
    //     conv_src_memory = memory(conv_prim_desc.src_desc(), eng);
    //     net.push_back(reorder(user_src_memory, conv_src_memory));
    //     net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
    //             {DNNL_ARG_TO, conv_src_memory}});
    // }

    auto conv_weights_memory = user_weights_memory;
    if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv_weights_memory = memory(conv_prim_desc.weights_desc(), eng);
        reorder(user_weights_memory, conv_weights_memory).execute(s, user_weights_memory, conv_weights_memory);
    }

    /// Create a memory primitive for output.
    auto conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);

    /// Create a convolution primitive and add it to the net.
    net.push_back(convolution_forward(conv_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, net_args.back()[DNNL_ARG_DST]},
                        {DNNL_ARG_WEIGHTS, conv_weights_memory},
                        {DNNL_ARG_BIAS, conv_user_bias_memory},
                        {DNNL_ARG_DST,  conv_dst_memory}});

    // relu1
    const float neg_slope = 1.0f;

    /// Create the relu primitive. For better performance, keep the input data
    /// format for ReLU (as well as for other operation primitives until another
    /// convolution or inner product is encountered) the same as the one chosen
    /// for convolution. Also note that ReLU is done in-place by using conv memory.
    auto relu_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, net_args.back()[DNNL_ARG_DST].get_desc(), neg_slope);
    auto relu_prim_desc = eltwise_forward::primitive_desc(relu_desc, eng);

    net.push_back(eltwise_forward(relu_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, net_args.back()[DNNL_ARG_DST]},
                        {DNNL_ARG_DST, net_args.back()[DNNL_ARG_DST]}});
}

void tomogan(engine::kind engine_kind, unsigned int img_sz){
    //                                   0    1   2   3   4   5    6    7    8    9   10   11  12  13  14  15
    const unsigned int conv_ch[16] = {3, 8,  32, 32, 64, 64,  128, 128, 256, 64, 128, 32, 64, 32, 32, 16};
    const unsigned int  n_conv[16] = {8, 32, 32, 64, 64, 128, 128, 128, 64,  64, 32,  32, 32, 32, 16, 1};
    const unsigned int conv_sz[16] = {1, 3,  3,  3,   3, 3,   3,   3,   3,   3,  3,   3,   3, 3,  1,  1};

    /// Initialize an engine and stream. The last parameter in the call represents
    /// the index of the engine.
    engine eng(engine_kind, 0);
    stream s(eng);
    const memory::dim batch = 1;

    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    memory::dims conv_src_tz = {batch, 3, img_sz, img_sz};
    /// Create memory that describes data layout in the buffers. here we use
    /// tag::nchw (batch-channels-height-width) for input data.
    auto user_src_memory = memory({{conv_src_tz}, memory::data_type::f32, memory::format_tag::nchw}, eng);
    /// Allocate buffers for input and output data, weights, and bias.
    std::vector<float> user_src(batch * conv_ch[0] * img_sz * img_sz);
    // copy input image to input buffer and then write to engine memory
    write_to_dnnl_memory(user_src.data(), user_src_memory);
    net_args.push_back({{DNNL_ARG_SRC, user_src_memory},
                        {DNNL_ARG_DST, user_src_memory}});

    conv2d(batch, conv_ch[0], conv_sz[0], n_conv[0], net, net_args, eng, s);
    conv2d(batch, conv_ch[1], conv_sz[1], n_conv[1], net, net_args, eng, s);
    conv2d(batch, conv_ch[2], conv_sz[2], n_conv[2], net, net_args, eng, s);

    std::cout << "desc: " << net_args.back()[DNNL_ARG_DST].get_desc().data.dims[2] << std::endl;
    for (auto args : net_args){
        auto in_dims = args[DNNL_ARG_SRC].get_desc().data.dims;
        printf("Input: %d x %3d x %d x %d", in_dims[0], in_dims[1], in_dims[2], in_dims[3]);

        auto out_dims = args[DNNL_ARG_DST].get_desc().data.dims;
        printf(" => Output: %d x %3d x %d x %d\n", out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    }

    auto mdl_exe_st = chrono::steady_clock::now();

    for (int j = 0; j < 10; ++j) {
        assert((net.size()+1) == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(s, net_args.at(i+1));
    }
    s.wait();

    auto mdl_exe_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to execute the model!\n", \
           chrono::duration_cast<chrono::microseconds>(mdl_exe_ed - mdl_exe_st).count()/1000.);
}