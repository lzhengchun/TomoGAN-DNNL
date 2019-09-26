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
    // std::vector<std::unordered_map<string, int>> net_args;
    // net_args.push_back({{"arg1", 11}, {"arg2", 12}, {"arg3", 13}});
    // net_args.push_back({{"arg1", 22}, {"arg2", 22}, {"arg3", 23}});
    // net_args.push_back({{"arg1", 31}, {"arg2", 32}, {"arg3", 33}});
    // std::cout << "last one: " << net_args.back()["arg1"] << std::endl;
    // for (auto args : net_args){
    //     std::cout << args["arg1"] << std::endl;
    //     for (auto arg : args){
    //         std::cout << arg.first  << ':' << arg.second << std::endl;
    //     }
    // }

    unsigned int img_sz = 1024;
    using tag = memory::format_tag;
    using dt  = memory::data_type;
    /// Initialize an engine and stream. The last parameter in the call represents
    /// the index of the engine.
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    stream s(eng);

    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    //[Create network]
    const memory::dim batch = 1;

    memory::dims conv_src_tz = {batch, 3, img_sz, img_sz};
    /// Create memory that describes data layout in the buffers. here we use
    /// tag::nchw (batch-channels-height-width) for input data.
    auto user_src_memory = memory({{conv_src_tz}, dt::f32, tag::nchw}, eng);
    /// Allocate buffers for input and output data, weights, and bias.
    std::vector<float> user_src(batch * 3 * img_sz * img_sz);
    // copy input image to input buffer and then write to engine memory
    write_to_dnnl_memory(user_src.data(), user_src_memory);
    net_args.push_back({{DNNL_ARG_DST, user_src_memory}});

    // std::cout << "total size in bytes: " << net_args.back()[DNNL_ARG_DST].get_desc().get_size() << std::endl;
    // CNN data tensors: mini-batch, channel, spatial ({N, C, [[D,] H,] W})
    // CNN weight tensors: group (optional), output channel, input channel, spatial ({[G,] O, I, [[D,] H,] W})
    std::cout << "desc: " << net_args.back()[DNNL_ARG_DST].get_desc().data.dims[2] << std::endl;
}
