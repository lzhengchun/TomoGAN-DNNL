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

void concatenate(memory::dim batch, 
    memory::dim tensor1_idx, 
    memory::dim tensor2_idx, 
    std::vector<primitive> &net,
    std::vector<std::unordered_map<int, memory>> &net_args,
    engine &eng, 
    stream &s){

    auto dim1 = net_args[tensor1_idx][DNNL_ARG_DST].get_desc().data.dims;
    auto dim2 = net_args[tensor2_idx][DNNL_ARG_DST].get_desc().data.dims;
    memory::dims concat_dst_tz  = {batch, dim1[1]+dim2[1], dim1[2], dim1[3]};
    std::vector<memory::desc> srcs_md;
    
    srcs_md.push_back(net_args[tensor1_idx][DNNL_ARG_DST].get_desc());
    srcs_md.push_back(net_args[tensor2_idx][DNNL_ARG_DST].get_desc());

    auto concat_dst_md = memory::desc({concat_dst_tz}, memory::data_type::f32, memory::format_tag::any);

    auto concat_pd = concat::primitive_desc(concat_dst_md, 1, srcs_md, eng);

    auto cat_dst_memory = memory(concat_pd.dst_desc(), eng);

    concat c(concat_pd);
    net.push_back(c);

    std::unordered_map<int, memory> args = {{DNNL_ARG_DST, cat_dst_memory}};
    args.insert({DNNL_ARG_MULTIPLE_SRC + 0, net_args[tensor1_idx][DNNL_ARG_DST]});
    args.insert({DNNL_ARG_MULTIPLE_SRC + 1, net_args[tensor2_idx][DNNL_ARG_DST]});

    net_args.push_back(args);
}

void maxpooling(memory::dim batch, 
    std::vector<primitive> &net,
    std::vector<std::unordered_map<int, memory>> &net_args,
    engine &eng, 
    stream &s){

    memory::dim in_img_c = net_args.back()[DNNL_ARG_DST].get_desc().data.dims[1];
    memory::dim in_img_h = net_args.back()[DNNL_ARG_DST].get_desc().data.dims[2];
    memory::dim in_img_w = net_args.back()[DNNL_ARG_DST].get_desc().data.dims[3];

    memory::dims pool_strides = {2, 2};
    memory::dims pool_kernel  = {2, 2};
    memory::dims pool_padding = {0, 0};

    memory::dims pool_dst_tz  = {batch, in_img_c, in_img_h/pool_strides[0], in_img_h/pool_strides[1]};

    auto pool_dst_md = memory::desc({pool_dst_tz}, memory::data_type::f32, memory::format_tag::any);

    auto pool_desc = pooling_forward::desc(prop_kind::forward_inference,
            algorithm::pooling_max, net_args.back()[DNNL_ARG_DST].get_desc(), pool_dst_md,
            pool_strides, pool_kernel, pool_padding, pool_padding);
    auto pool_pd = pooling_forward::primitive_desc(pool_desc, eng);
    auto pool_dst_memory = memory(pool_pd.dst_desc(), eng);

    net.push_back(pooling_forward(pool_pd));
    net_args.push_back({{DNNL_ARG_SRC, net_args.back()[DNNL_ARG_DST]},
                        {DNNL_ARG_DST, pool_dst_memory}});
}

void conv2d(memory::dim batch, 
    memory::dim in_ch,
    memory::dim knl_sz,
    memory::dim n_knl,
    std::vector<primitive> &net,
    std::vector<std::unordered_map<int, memory>> &net_args,
    engine &eng, 
    stream &s,
    bool has_relu){
    
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
    /// from `nchw`, the user data must be transformed to the format required for the convolution.
    auto conv_bias_md    = memory::desc({conv_bias_tz},    memory::data_type::f32, memory::format_tag::any);
    auto conv_weights_md = memory::desc({conv_weights_tz}, memory::data_type::f32, memory::format_tag::any);
    auto conv_dst_md     = memory::desc({conv_dst_tz},     memory::data_type::f32, memory::format_tag::any);

    /// Create a convolution descriptor
    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_auto, net_args.back()[DNNL_ARG_DST].get_desc(), conv_weights_md,
            conv_bias_md, conv_dst_md, conv_strides, conv_padding, conv_padding);

    /// Create a convolution primitive descriptor. Once created, this
    /// descriptor has specific formats instead of the `any` format specified
    /// in the convolution descriptor.
    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);
    
    /// Create a memory primitive for output.
    auto conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);

    /// Create a convolution primitive and add it to the net.
    net.push_back(convolution_forward(conv_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, net_args.back()[DNNL_ARG_DST]},
                        {DNNL_ARG_WEIGHTS, user_weights_memory},
                        {DNNL_ARG_BIAS, conv_user_bias_memory},
                        {DNNL_ARG_DST,  conv_dst_memory}});
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
    /// Allocate buffers for input and output data, weights, and bias.
    std::vector<float> user_src(batch * conv_ch[0] * img_sz * img_sz);
    // copy input image to input buffer and then write to engine memory
    write_to_dnnl_memory(user_src.data(), user_src_memory);
    net_args.push_back({{DNNL_ARG_SRC, user_src_memory},
                        {DNNL_ARG_DST, user_src_memory}});

    conv2d(batch, conv_ch[0], conv_sz[0], n_conv[0], net, net_args, eng, s, true);
    conv2d(batch, conv_ch[1], conv_sz[1], n_conv[1], net, net_args, eng, s, true);
    conv2d(batch, conv_ch[2], conv_sz[2], n_conv[2], net, net_args, eng, s, true);
    maxpooling(batch, net, net_args, eng, s);
    conv2d(batch, conv_ch[3], conv_sz[3], n_conv[3], net, net_args, eng, s, true);
    conv2d(batch, conv_ch[4], conv_sz[4], n_conv[4], net, net_args, eng, s, true);
    maxpooling(batch, net, net_args, eng, s);
    conv2d(batch, conv_ch[5], conv_sz[5], n_conv[5], net, net_args, eng, s, true);
    conv2d(batch, conv_ch[6], conv_sz[6], n_conv[6], net, net_args, eng, s, true);
    maxpooling(batch, net, net_args, eng, s);
    conv2d(batch, conv_ch[7], conv_sz[7], n_conv[7], net, net_args, eng, s, true);

    // concatenate(batch, 18, 19, net, net_args, eng, s);

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

    unsigned int conv_idx_in_netarg[] = {1, 3, 5, 8, 10, 13, 15, 18};
    std::ifstream weights_fin("tomogan_weights_serilize-oihw.bin", std::ios::binary);
    for(int i = 0; i < 8; i++){
        unsigned int n_weights = (conv_sz[i] * conv_sz[i] * conv_ch[i] * n_conv[i]);
        unsigned int wbuf_size = sizeof(float) * n_weights;
        printf("%6d paras (%6d Bytes) for conv2d_%02d kernel in_ch: %3d, no_ch: %3d\n", n_weights, wbuf_size, i, conv_ch[i], n_conv[i]);
        auto weights_buf = new float[wbuf_size]();
        auto bias_buf    = new float[n_conv[i]]();
        weights_fin.read((char *) weights_buf, wbuf_size);
        if(weights_fin){
            auto in_dims = net_args[conv_idx_in_netarg[i]][DNNL_ARG_WEIGHTS].get_desc().data.dims;
            printf("conv layer %d Weights: %d x %d x %d x %d\n", i, in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
            write_to_dnnl_memory(weights_buf, net_args[conv_idx_in_netarg[i]][DNNL_ARG_WEIGHTS]);
            write_to_dnnl_memory(bias_buf,    net_args[conv_idx_in_netarg[i]][DNNL_ARG_BIAS]);
        }else{
            printf("Error while load weights for conv %02d, EoF reached, only %ld bytes could be read\n", i, weights_fin.gcount());
            exit(-1);
        }
        delete[] weights_buf;
        delete[] bias_buf;
    }
    weights_fin.close();

    // load input data
    auto input_buf = new float[img_sz * img_sz * 3]();
    std::ifstream inputs_fin("test_input_serilize-nchw.bin", std::ios::binary);
    inputs_fin.read((char *)input_buf, sizeof(float) * img_sz * img_sz * 3);
    if(inputs_fin){
        printf("%ld bytes of input data have been successfully read\n", inputs_fin.gcount());
        write_to_dnnl_memory(input_buf, net_args[0][DNNL_ARG_DST]);
    }else{
        printf("Error while load input, EoF reached, only %ld bytes could be read\n", inputs_fin.gcount());
        exit(-1);
    }
    inputs_fin.close();

    std::cout << "Model built, executing now ..." << std::endl;

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

    auto output_dims = net_args.back()[DNNL_ARG_DST].get_desc().data.dims;
    auto output_size = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
    auto output_buf = new float[output_size]();
    read_from_dnnl_memory(output_buf, net_args.back()[DNNL_ARG_DST]);
    // dump output array to a file
    std::ofstream img_fout("output_img.bin", std::ios::out | std::ios::binary);
    img_fout.write((char *) output_buf, sizeof(float) * output_size);
    img_fout.close();
    delete[] output_buf;
}