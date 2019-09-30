#include <assert.h>
#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include "tomogan.hpp"

using namespace std;
using namespace dnnl;

void conv2d_opt( 
    memory::dim knl_sz,
    memory::dim n_knl,
    std::vector<primitive> &net,
    std::vector<std::unordered_map<int, memory>> &net_args,
    engine &eng, 
    stream &s,
    bool has_relu){
    
    auto in_dims = net_args.back()[DNNL_ARG_DST].get_desc().data.dims;
    memory::dim batch    = in_dims[0];
    memory::dim in_img_c = in_dims[1];
    memory::dim in_img_h = in_dims[2];
    memory::dim in_img_w = in_dims[3];

    memory::dims conv_src_tz = {batch, in_img_c, in_img_h, in_img_w};
    memory::dims conv_weights_tz = {n_knl, in_img_c, knl_sz, knl_sz};
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
    /// from `nchw`, the user data must be transformed to the format required for the convolution.
    auto conv_src_md     = memory::desc({conv_src_tz},     memory::data_type::f32, memory::format_tag::any);
    auto conv_bias_md    = memory::desc({conv_bias_tz},    memory::data_type::f32, memory::format_tag::x);
    auto conv_weights_md = memory::desc({conv_weights_tz}, memory::data_type::f32, memory::format_tag::any);
    auto conv_dst_md     = memory::desc({conv_dst_tz},     memory::data_type::f32, memory::format_tag::any);

    auto user_src_memory = net_args.back()[DNNL_ARG_DST];
    /// Create a convolution descriptor
    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_auto, conv_src_md, conv_weights_md,
            conv_bias_md, conv_dst_md, conv_strides, conv_padding, conv_padding);

    /// Create a convolution primitive descriptor. Once created, this
    /// descriptor has specific formats instead of the 'any' format specified
    /// in the convolution descriptor.
    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);
    
    bool need_reorder_src = conv_prim_desc.src_desc() != user_src_memory.get_desc();
    auto conv_src_memory = need_reorder_src ? memory(conv_prim_desc.src_desc(), eng) : user_src_memory;
    if (need_reorder_src) {
        std::cout << "!!!!! src mem needs reorder" << std::endl;
        // reorder(user_src_memory, conv_src_memory).execute(s, 
        //         {{DNNL_ARG_FROM, user_src_memory}, {DNNL_ARG_TO, conv_src_memory}});
        // s.wait();
        net.push_back(reorder(user_src_memory, conv_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                            {DNNL_ARG_TO, conv_src_memory}});
    }

    bool need_reorder_weights = conv_prim_desc.weights_desc() != user_weights_memory.get_desc();
    auto conv_weights_memory  = need_reorder_weights ? memory(conv_prim_desc.weights_desc(), eng) : user_weights_memory;
    if (need_reorder_weights) {
        std::cout << "!!!!! weight mem needs reorder" << std::endl;
        // reorder(user_weights_memory, conv_weights_memory).execute(s, 
        //     {{DNNL_ARG_FROM, user_weights_memory}, {DNNL_ARG_TO, conv_weights_memory}});
        // s.wait();
        net.push_back(reorder(user_weights_memory, conv_weights_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_weights_memory},
                            {DNNL_ARG_TO, conv_weights_memory}});
    }

    /// Create a memory primitive for output.
    auto conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);

    /// Create a convolution primitive and add it to the net.
    net.push_back(convolution_forward(conv_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC,     conv_src_memory},
                        {DNNL_ARG_WEIGHTS, conv_weights_memory},
                        {DNNL_ARG_WEIGHTS_CUS, user_weights_memory},
                        {DNNL_ARG_BIAS,    conv_user_bias_memory},
                        {DNNL_ARG_DST,     conv_dst_memory}});
    // relu
    if(has_relu){
        /// Create the relu primitive. For better performance, keep the input data
        /// format for ReLU (as well as for other operation primitives until another
        /// convolution or inner product is encountered) the same as the one chosen
        /// for convolution. Also note that ReLU is done in-place by using conv memory.
        auto relu_desc = eltwise_forward::desc(prop_kind::forward_inference,
                algorithm::eltwise_relu, net_args.back()[DNNL_ARG_DST].get_desc(), 1.0f);
        auto relu_prim_desc = eltwise_forward::primitive_desc(relu_desc, eng);

        net.push_back(eltwise_forward(relu_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, net_args.back()[DNNL_ARG_DST]},
                            {DNNL_ARG_DST, net_args.back()[DNNL_ARG_DST]}});
    }
}

int main(int argc, char const *argv[]){
    /// Initialize an engine and stream. 
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    stream s(eng);
    const memory::dim batch = 1;
    const memory::dim img_sz = 1024;
    const memory::dim img_ch = 3;
    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    memory::dims conv_src_tz = {batch, img_ch, img_sz, img_sz};
    /// Create memory that describes data layout in the buffers. here we use
    /// tag::nchw for input data.
    auto user_src_memory = memory({{conv_src_tz}, memory::data_type::f32, memory::format_tag::nchw}, eng);
    /// Allocate buffers for input and output data, weights, and bias.
    std::vector<float> user_src(batch * img_ch * img_sz * img_sz);
    // copy input image to input buffer and then write to engine memory
    write_to_dnnl_memory(user_src.data(), user_src_memory);
    net_args.push_back({{DNNL_ARG_SRC, user_src_memory},
                        {DNNL_ARG_DST, user_src_memory}});

    conv2d_opt(3, 8, net, net_args, eng, s, true);
    conv2d_opt(3, 32, net, net_args, eng, s, true);
    
    for (auto args : net_args){
        if(args.find(DNNL_ARG_SRC) != args.end()){
            auto in_dims = args[DNNL_ARG_SRC].get_desc().data.dims;
            printf("Input: %d x %3d x %d x %d", in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
        }else{
            auto in_dims = args[DNNL_ARG_MULTIPLE_SRC + 0].get_desc().data.dims;
            printf("Input: %d x %3d x %d x %d, ", in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
            in_dims = args[DNNL_ARG_MULTIPLE_SRC + 1].get_desc().data.dims;
            printf("%d x %3d x %d x %d", in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
        }

        auto out_dims = args[DNNL_ARG_DST].get_desc().data.dims;
        printf(" => Output: %d x %3d x %d x %d\n", out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    }

    // set all weights to 1 and bias to zero for debug
    for(int i = 0; i < net_args.size(); i++){
        if(net_args[i].find(DNNL_ARG_WEIGHTS) == net_args[i].end()){
            continue;
        }

        auto in_dims = net_args[i][DNNL_ARG_WEIGHTS_CUS].get_desc().data.dims;
        unsigned int wbuf_size = in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3];

        std::vector<float> conv_weights(wbuf_size);
        for(auto i = 0; i < conv_weights.size(); i++){
            conv_weights[i] = 1;
        }
        
        std::vector<float> conv_bias(in_dims[0]);
        for(auto i = 0; i < conv_bias.size(); i++){
            conv_bias[i] = 0;
        }
        
        printf("conv layer %d Weights: %d x %d x %d x %d\n", i, in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
        write_to_dnnl_memory(conv_weights.data(), net_args[i][DNNL_ARG_WEIGHTS_CUS]);
        write_to_dnnl_memory(conv_bias.data(),    net_args[i][DNNL_ARG_BIAS]);
    }
    // explicitly set input value for debug
    auto in_dims = net_args[0][DNNL_ARG_DST].get_desc().data.dims;
    unsigned int in_buf_size = in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3];
    std::vector<float> input_buf(in_buf_size);
    for(auto n = 0; n < in_dims[0]; n++)
        for(auto c = 0; c < in_dims[1]; c++)
            for(auto h = 0; h < in_dims[2]; h++)
                for(auto w = 0; w < in_dims[3]; w++){
                    input_buf[n * in_dims[1] * in_dims[2] * in_dims[3] +\
                              c * in_dims[2] * in_dims[3] + \
                              h * in_dims[3] + w] = c;
                }

    write_to_dnnl_memory(input_buf.data(), net_args[0][DNNL_ARG_DST]);

    auto mdl_exe_st = chrono::steady_clock::now();

    assert((net.size()+1) == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i)
        net.at(i).execute(s, net_args.at(i+1));
    s.wait();
    
    auto mdl_exe_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to execute the model!\n", \
           chrono::duration_cast<chrono::microseconds>(mdl_exe_ed - mdl_exe_st).count()/1000.);

    // <should check if reorder needed> copy results from dnnl device to cpu mempry
    auto output_dims = net_args.back()[DNNL_ARG_DST].get_desc().data.dims;
    auto output_size = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
    auto output_buf = new float[output_size]();
    read_from_dnnl_memory(output_buf, net_args.back()[DNNL_ARG_DST]);

    double sum = 0;
    for(size_t i = 0; i < output_size; i++){
        sum += output_buf[i];
    }
    printf("results checksum: %lf\n", sum);

    delete[] output_buf;
}