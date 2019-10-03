#include <assert.h>
#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <iostream>
#include <fstream>
#include <string>

#include "tomogan.hpp"

using namespace std;
using namespace dnnl;

void conv2d_transpose( 
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

    memory::dims deconv_src_tz = {batch, in_img_c, in_img_h, in_img_w};
    memory::dims deconv_weights_tz = {n_knl, in_img_c, knl_sz, knl_sz};
    memory::dims deconv_bias_tz = {n_knl};
    memory::dims deconv_dst_tz = {batch, n_knl, in_img_h*2, in_img_w*2};
    memory::dims deconv_strides = {2, 2};
    memory::dims deconv_padding = {0, 0}; // padding as the same

    /// Create memory that describes data layout in the buffers. here we use
    /// tag::oihw for weights.
    auto user_weights_memory     = memory({{deconv_weights_tz}, memory::data_type::f32, memory::format_tag::oihw}, eng);
    // auto deconv_user_bias_memory = memory({{deconv_bias_tz},    memory::data_type::f32, memory::format_tag::x},    eng);
    
    auto deconv_src_md     = memory::desc({deconv_src_tz},     memory::data_type::f32, memory::format_tag::any);
    // auto deconv_bias_md    = memory::desc({deconv_bias_tz},    memory::data_type::f32, memory::format_tag::x);
    auto deconv_weights_md = memory::desc({deconv_weights_tz}, memory::data_type::f32, memory::format_tag::any);
    auto deconv_dst_md     = memory::desc({deconv_dst_tz},     memory::data_type::f32, memory::format_tag::any);

    auto user_src_memory = net_args.back()[DNNL_ARG_DST];
    /// Create a convolution descriptor
    auto deconv_desc = convolution_backward_data::desc(
            algorithm::convolution_auto, deconv_src_md, deconv_weights_md,
            deconv_dst_md, deconv_strides, deconv_padding, deconv_padding);

    /// Create a convolution primitive descriptor. Once created, this
    /// descriptor has specific formats instead of the 'any' format specified
    /// in the convolution descriptor.
    auto deconv_prim_desc = convolution_backward_data::primitive_desc(deconv_desc, eng, nullptr, true);
    
    bool need_reorder_src = deconv_prim_desc.src_desc() != user_src_memory.get_desc();
    auto deconv_src_memory = need_reorder_src ? memory(deconv_prim_desc.src_desc(), eng) : user_src_memory;
    if (need_reorder_src) {
        std::cout << "!!!!! src mem needs reorder for deconv" << std::endl;
        net.push_back(reorder(user_src_memory, deconv_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                            {DNNL_ARG_TO, deconv_src_memory}});
    }

    bool need_reorder_weights = deconv_prim_desc.weights_desc() != user_weights_memory.get_desc();
    auto deconv_weights_memory  = need_reorder_weights ? memory(deconv_prim_desc.weights_desc(), eng) : user_weights_memory;
    if (need_reorder_weights) {
        std::cout << "!!!!! weight mem needs reorder for deconv" << std::endl;
        net.push_back(reorder(user_weights_memory, deconv_weights_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_weights_memory},
                            {DNNL_ARG_TO, deconv_weights_memory}});
    }

    /// Create a memory primitive for output.
    auto deconv_dst_memory = memory(deconv_prim_desc.dst_desc(), eng);
    
    /// Create a convolution primitive and add it to the net.
    net.push_back(deconvolution_forward(deconv_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC,     deconv_src_memory},
                        {DNNL_ARG_WEIGHTS, deconv_weights_memory},
                        {DNNL_ARG_WEIGHTS_CUS, user_weights_memory},
                        // {DNNL_ARG_BIAS,    deconv_user_bias_memory},
                        {DNNL_ARG_DST,     deconv_dst_memory}});

    // relu, does not seem to support fusion
    if(has_relu){
        auto relu_desc = eltwise_forward::desc(prop_kind::forward_inference,
                algorithm::eltwise_relu, net_args.back()[DNNL_ARG_DST].get_desc(), 0.0f);
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
    const memory::dim img_sz = 128;
    const memory::dim img_ch = 128;
    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    memory::dims conv_src_tz = {batch, img_ch, img_sz, img_sz};
    /// Create memory that describes data layout in the buffers. here we use tag::nchw for input data.
    auto user_src_memory = memory({{conv_src_tz}, memory::data_type::f32, memory::format_tag::nchw}, eng);
    // load input data
    auto input_buf_size = batch * img_ch * img_sz * img_sz;
    auto input_buf = new float[input_buf_size]();
    std::ifstream inputs_fin("dsp-out-nchw.bin", std::ios::binary);
    inputs_fin.read((char *)input_buf, sizeof(float) * input_buf_size);
    if(inputs_fin){
        printf("%ld bytes of input data have been successfully read\n", inputs_fin.gcount());
        write_to_dnnl_memory(input_buf, user_src_memory);
    }else{
        printf("Error while load input, EoF reached, only %ld bytes could be read\n", inputs_fin.gcount());
        exit(-1);
    }
    inputs_fin.close();
    delete[] input_buf;
    // input tensor
    net_args.push_back({{DNNL_ARG_SRC, user_src_memory},
                        {DNNL_ARG_DST, user_src_memory}});

    conv2d_transpose(2, 128, net, net_args, eng, s, true);
    
    // reorder output before copy back to make sure it is NCHW in user space
    auto out_dims = net_args.back()[DNNL_ARG_DST].get_desc().data.dims;
    auto out_md   = memory::desc({out_dims[0], out_dims[1], out_dims[2], out_dims[3]}, \
                                 memory::data_type::f32, memory::format_tag::nchw);
    if(net_args.back()[DNNL_ARG_DST].get_desc() != out_md){
        auto dst_mem = memory(out_md, eng);
        net.push_back(reorder(net_args.back()[DNNL_ARG_DST], dst_mem));
        net_args.push_back({{DNNL_ARG_FROM, net_args.back()[DNNL_ARG_DST]},
                            {DNNL_ARG_TO, dst_mem}});
    }
    // print node in/out shape
    for (auto args : net_args){
        if(args.find(DNNL_ARG_SRC) != args.end()){
            auto in_dims = args[DNNL_ARG_SRC].get_desc().data.dims;
            printf("Input: %ld x %3ld x %ld x %ld", in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
        }else{
            auto in_dims = args[DNNL_ARG_MULTIPLE_SRC + 0].get_desc().data.dims;
            printf("Input: %ld x %3ld x %ld x %ld, ", in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
            in_dims = args[DNNL_ARG_MULTIPLE_SRC + 1].get_desc().data.dims;
            printf("%ld x %3ld x %ld x %ld", in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
        }

        auto out_dims = args[DNNL_ARG_DST].get_desc().data.dims;
        printf(" => Output: %ld x %3ld x %ld x %ld\n", out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    }

    // load weights
    int total_weights = 0;
    std::ifstream weights_fin("deconv-oihw.bin", std::ios::binary);
    auto deconv_node_idx = 2;
    if(net_args[deconv_node_idx].find(DNNL_ARG_WEIGHTS_CUS) == net_args[deconv_node_idx].end()){
        printf("node %d is not a deconv operation\n", deconv_node_idx);
        exit(-1);
    }
    auto in_dims = net_args[deconv_node_idx][DNNL_ARG_WEIGHTS_CUS].get_desc().data.dims;
    unsigned int wbuf_size = in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3];

    std::vector<float> conv_weights(wbuf_size);
    weights_fin.read((char *) conv_weights.data(), sizeof(float) * wbuf_size);
    
    std::vector<float> conv_bias(in_dims[0]); // weights layout is oihw
    weights_fin.read((char *) conv_bias.data(), sizeof(float) * in_dims[0]);

    if(weights_fin){
        printf("load Weights: %ld x %ld x %ld x %ld\n", in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
        write_to_dnnl_memory(conv_weights.data(), net_args[deconv_node_idx][DNNL_ARG_WEIGHTS_CUS]);
        write_to_dnnl_memory(conv_bias.data(),    net_args[deconv_node_idx][DNNL_ARG_BIAS]);
        total_weights += (in_dims[0] + wbuf_size);
    }else{
        printf("Error while load weights EoF reached, only %ld bytes could be read\n", weights_fin.gcount());
        exit(-1);
    }
    weights_fin.close();
    printf("%d weights loaded!\n", total_weights);

    assert((net.size()+1) == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i)
        net.at(i).execute(s, net_args.at(i+1));
    s.wait();
    
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
