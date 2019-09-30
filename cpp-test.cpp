#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <unordered_map>
#include "tomogan.hpp"
#include "dnnl.hpp"

using namespace std;
using namespace dnnl;

int main(int argc, char const *argv[]){
    unsigned int img_sz = 1024;
    using tag = memory::format_tag;
    using dt  = memory::data_type;
    /// Initialize an engine and stream. The last parameter in the call represents
    /// the index of the engine.
    dnnl::engine gpu_eng(dnnl::engine::kind::gpu, 0);
    dnnl::engine cpu_eng(dnnl::engine::kind::cpu, 0);
    stream s(gpu_eng);

    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    const memory::dim batch = 1;

    memory::dims conv_src_tz = {batch, img_sz, img_sz, 3};
    auto user_src_memory = memory({{conv_src_tz}, dt::f32, tag::nhwc}, gpu_eng);

    std::vector<float> user_src(batch * 3 * img_sz * img_sz);
    write_to_dnnl_memory(user_src.data(), user_src_memory);

    memory::dims conv_dst_tz = {batch, img_sz, img_sz, 3};
    auto user_dst_memory = memory({{conv_dst_tz}, dt::f32, tag::nchw}, cpu_eng);

    // if(user_src_memory.get_desc() != user_dst_memory.get_desc()){
    //     std::cout << "need reorder" << std::endl;
    //     net.push_back(reorder(user_src_memory, user_dst_memory));
    //     net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
    //                         {DNNL_ARG_TO,   user_dst_memory}});
    // }else{
    //     std::cout << "the same mem desc, no reorder" << std::endl;
    // }

    // net.at(0).execute(s, net_args.at(0));
    // s.wait();
    user_dst_memory.reshape({batch, 3, img_sz, img_sz});
    net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                        {DNNL_ARG_TO,   user_dst_memory}});

    auto output_dims = net_args.back()[DNNL_ARG_TO].get_desc().data.dims;
    printf("%d x %3d x %d x %d\n", output_dims[0], output_dims[1], output_dims[2], output_dims[3]);

}
