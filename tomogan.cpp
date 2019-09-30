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

memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
            std::multiplies<memory::dim>());
}

void concatenate(
    memory::dim tensor1_idx, 
    memory::dim tensor2_idx, 
    memory::dim cat_axis,
    std::vector<primitive> &net,
    std::vector<std::unordered_map<int, memory>> &net_args,
    engine &eng, 
    stream &s){

    auto dim1 = net_args[tensor1_idx][DNNL_ARG_DST].get_desc().data.dims;
    auto dim2 = net_args[tensor2_idx][DNNL_ARG_DST].get_desc().data.dims;

    memory::dim batch = dim1[0];

    memory::dims concat_dst_tz  = {batch, dim1[1]+dim2[1], dim1[2], dim1[3]};
    std::vector<memory::desc> srcs_md;
    
    srcs_md.push_back(net_args[tensor1_idx][DNNL_ARG_DST].get_desc());
    srcs_md.push_back(net_args[tensor2_idx][DNNL_ARG_DST].get_desc());

    auto concat_dst_md = memory::desc({concat_dst_tz}, memory::data_type::f32, memory::format_tag::any);

    auto concat_pd = concat::primitive_desc(concat_dst_md, cat_axis, srcs_md, eng);

    auto cat_dst_memory = memory(concat_pd.dst_desc(), eng);

    concat c(concat_pd);
    net.push_back(c);

    std::unordered_map<int, memory> args = {{DNNL_ARG_DST, cat_dst_memory}};
    args.insert({DNNL_ARG_MULTIPLE_SRC + 0, net_args[tensor1_idx][DNNL_ARG_DST]});
    args.insert({DNNL_ARG_MULTIPLE_SRC + 1, net_args[tensor2_idx][DNNL_ARG_DST]});

    net_args.push_back(args);
}

void maxpooling(
    std::vector<primitive> &net,
    std::vector<std::unordered_map<int, memory>> &net_args,
    engine &eng, 
    stream &s){

    auto in_dims = net_args.back()[DNNL_ARG_DST].get_desc().data.dims;
    memory::dim batch    = in_dims[0];
    memory::dim in_img_c = in_dims[1];
    memory::dim in_img_h = in_dims[2];
    memory::dim in_img_w = in_dims[3];

    memory::dims pool_strides = {2, 2};
    memory::dims pool_kernel  = {2, 2};
    memory::dims pool_padding = {0, 0};

    memory::dims pool_dst_tz  = {batch, in_img_c, in_img_h/pool_strides[0], in_img_h/pool_strides[1]};

    auto pool_dst_md = memory::desc({pool_dst_tz}, memory::data_type::f32, memory::format_tag::any);

    auto pool_desc = pooling_forward::desc(prop_kind::forward_inference,
            algorithm::pooling_max, net_args.back()[DNNL_ARG_DST].get_desc(), pool_dst_md,
            pool_strides, pool_kernel, pool_padding, pool_padding);
    auto pool_pd = pooling_forward::primitive_desc(pool_desc, eng);

    auto user_src_memory = net_args.back()[DNNL_ARG_DST];

    bool need_reorder_src = pool_pd.src_desc() != user_src_memory.get_desc();
    auto pool_src_memory = need_reorder_src ? memory(pool_pd.src_desc(), eng) : user_src_memory;
    if (need_reorder_src) {
        std::cout << "!!!!! src mem needs reorder" << std::endl;
        reorder(user_src_memory, pool_src_memory).execute(s, 
                {{DNNL_ARG_FROM, user_src_memory}, {DNNL_ARG_TO, pool_src_memory}});
        s.wait();
        // net.push_back(reorder(user_src_memory, pool_src_memory));
        // net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
        //                     {DNNL_ARG_TO, pool_src_memory}});
    }

    auto pool_dst_memory = memory(pool_pd.dst_desc(), eng);

    net.push_back(pooling_forward(pool_pd));
    net_args.push_back({{DNNL_ARG_SRC, pool_src_memory},
                        {DNNL_ARG_DST, pool_dst_memory}});
}

void conv2d( 
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
    convolution_forward::primitive_desc conv_prim_desc;
    if(has_relu){
        // create a post-op with relu
        post_ops relu_ops;
        relu_ops.append_eltwise(1.f, algorithm::eltwise_relu, 0.f, 0.f);
        // create an attribute and set the corresponding post op
        primitive_attr attr;
        attr.set_post_ops(relu_ops);
        conv_prim_desc = convolution_forward::primitive_desc(conv_desc, attr, eng);
    }else{
        conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);
    }
    
    bool need_reorder_src = conv_prim_desc.src_desc() != user_src_memory.get_desc();
    auto conv_src_memory = need_reorder_src ? memory(conv_prim_desc.src_desc(), eng) : user_src_memory;
    if (need_reorder_src) {
        std::cout << "!!!!! src mem needs reorder" << std::endl;
        net.push_back(reorder(user_src_memory, conv_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                            {DNNL_ARG_TO, conv_src_memory}});
    }

    bool need_reorder_weights = conv_prim_desc.weights_desc() != user_weights_memory.get_desc();
    auto conv_weights_memory  = need_reorder_weights ? memory(conv_prim_desc.weights_desc(), eng) : user_weights_memory;
    if (need_reorder_weights) {
        std::cout << "!!!!! weight mem needs reorder" << std::endl;
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
 }

void tomogan(engine::kind engine_kind, unsigned int img_sz){
    //                                0    1   2   3   4   5    6    7    8    9   10   11  12  13  14  15
    const unsigned int conv_ch[16] = {3, 8,  32, 32, 64, 64,  128, 128, 256, 64, 128, 32, 64, 32, 32, 16};
    const unsigned int  n_conv[16] = {8, 32, 32, 64, 64, 128, 128, 128, 64,  64, 32,  32, 32, 32, 16, 1};
    const unsigned int conv_sz[16] = {1, 3,  3,  3,   3, 3,   3,   3,   3,   3,  3,   3,   3, 3,  1,  1};

    /// Initialize an engine and stream. 
    engine eng(engine_kind, 0);
    stream s(eng);
    const memory::dim batch = 1;

    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    memory::dims conv_src_tz = {batch, 3, img_sz, img_sz};
    /// Create memory that describes data layout in the buffers. here we use
    /// tag::nchw for input data.
    auto user_src_memory = memory({{conv_src_tz}, memory::data_type::f32, memory::format_tag::nchw}, eng);
    // load input data
    auto input_buf = new float[batch * 3 * img_sz * img_sz]();
    std::ifstream inputs_fin("test_input_serilize-nchw.bin", std::ios::binary);
    inputs_fin.read((char *)input_buf, sizeof(float) * img_sz * img_sz * 3);
    if(inputs_fin){
        printf("%ld bytes of input data have been successfully read\n", inputs_fin.gcount());
        write_to_dnnl_memory(input_buf, user_src_memory);
    }else{
        printf("Error while load input, EoF reached, only %ld bytes could be read\n", inputs_fin.gcount());
        exit(-1);
    }
    inputs_fin.close();

    net_args.push_back({{DNNL_ARG_SRC, user_src_memory}, {DNNL_ARG_DST, user_src_memory}});

    conv2d(conv_sz[0], n_conv[0], net, net_args, eng, s, true);
    conv2d(conv_sz[1], n_conv[1], net, net_args, eng, s, true);
    conv2d(conv_sz[2], n_conv[2], net, net_args, eng, s, true);
    maxpooling(net, net_args, eng, s);
    conv2d(conv_sz[3], n_conv[3], net, net_args, eng, s, true);
    conv2d(conv_sz[4], n_conv[4], net, net_args, eng, s, true);
    maxpooling(net, net_args, eng, s);
    conv2d(conv_sz[5], n_conv[5], net, net_args, eng, s, true);
    conv2d(conv_sz[6], n_conv[6], net, net_args, eng, s, true);
    maxpooling(net, net_args, eng, s);
    conv2d(conv_sz[7], n_conv[7], net, net_args, eng, s, true);

    // concatenate(18, 19, 1, net, net_args, eng, s);

    // reorder output before copy back to make sure it is NCHW in user space
    auto out_dims = net_args.back()[DNNL_ARG_DST].get_desc().data.dims;
    auto dst_mem = memory({{out_dims[0], out_dims[1], out_dims[2], out_dims[3]}, \
                            memory::data_type::f32, memory::format_tag::nchw}, eng);
    if(net_args.back()[DNNL_ARG_DST].get_desc() != dst_mem.get_desc()){
        std::cout << "!!!!! output needs reorder for user" << std::endl;
        net.push_back(reorder(net_args.back()[DNNL_ARG_DST], dst_mem));
        net_args.push_back({{DNNL_ARG_FROM, net_args.back()[DNNL_ARG_DST]},
                            {DNNL_ARG_TO, dst_mem}});
    }
    
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

    std::ifstream weights_fin("tomogan_weights_serilize-oihw.bin", std::ios::binary);
    for(int i = 0; i < net_args.size(); i++){
        if(net_args[i].find(DNNL_ARG_WEIGHTS) == net_args[i].end()){
            continue;
        }

        auto in_dims = net_args[i][DNNL_ARG_WEIGHTS].get_desc().data.dims;
        unsigned int wbuf_size = in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3];

        std::vector<float> conv_weights(wbuf_size);
        weights_fin.read((char *) conv_weights.data(), sizeof(float) * wbuf_size);
        
        std::vector<float> conv_bias(in_dims[0], 0);
        
        if(weights_fin){
            printf("conv layer %d Weights: %d x %d x %d x %d\n", i, in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
            write_to_dnnl_memory(conv_weights.data(), net_args[i][DNNL_ARG_WEIGHTS_CUS]);
            write_to_dnnl_memory(conv_bias.data(),    net_args[i][DNNL_ARG_BIAS]);
        }else{
            printf("Error while load weights for conv %02d, EoF reached, only %ld bytes could be read\n", i, weights_fin.gcount());
            exit(-1);
        }
    }

    weights_fin.close();

    std::cout << "Model built, executing now ..." << std::endl;

    // warm up
    assert((net.size()+1) == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i)
        net.at(i).execute(s, net_args.at(i+1));
    s.wait();

    auto mdl_exe_st = chrono::steady_clock::now();

    size_t rep_times = 1;
    for (size_t i = 0; i < rep_times; i++){
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(s, net_args.at(i+1));
        s.wait();
    }
    
    auto mdl_exe_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to execute the model!\n", \
           chrono::duration_cast<chrono::microseconds>(mdl_exe_ed - mdl_exe_st).count()/1000./rep_times);

    // copy results from dnnl device to cpu mempry
    auto output_dims = net_args.back()[DNNL_ARG_DST].get_desc().data.dims;
    auto output_size = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
    auto output_buf = new float[output_size]();
    read_from_dnnl_memory(output_buf, net_args.back()[DNNL_ARG_DST]);

    double sum = 0;
    for(size_t i = 0; i < output_size; i++){
        sum += output_buf[i];
    }
    printf("results checksum: %lf\n", sum);

    // dump output array to a file
    std::ofstream img_fout("output_img.bin", std::ios::out | std::ios::binary);
    img_fout.write((char *) output_buf, sizeof(float) * output_size);
    img_fout.close();
    delete[] output_buf;
}