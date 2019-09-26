#include <iostream>
#include <cstdint>
#include <cstring>

void upsample_cpu(float *input,
                  const unsigned int height,
                  const unsigned int width,
                  const unsigned int channel,
                  float    *output){
    for(size_t r = 0; r < height; r++)
        for(size_t c = 0; c < width; c++){
            unsigned int urow = 2 * r;
            unsigned int ucol = 2 * c;
            unsigned int uwidth = 2 * width;
            unsigned int uheight= 2 * height;
            for(unsigned int ch = 0; ch < channel; ch++){
                float pixel = input[width * r * channel + c * channel + ch];
                output[uwidth * urow     * channel + ucol     * channel + ch] = pixel;  // [urow][ucol][ch] 
                output[uwidth * (urow+1) * channel + ucol     * channel + ch] = pixel;  // [urow+1][ucol][ch] 
                output[uwidth * urow     * channel + (ucol+1) * channel + ch] = pixel;  // [urow][ucol+1][ch]
                output[uwidth * (urow+1) * channel + (ucol+1) * channel + ch] = pixel;  // [urow+1][ucol+1][ch]
            }

        }

}

void concatenate(float *input1,
                 float *input2,
                 unsigned int height,
                 unsigned int width,
                 unsigned int channel1,
                 unsigned int channel2,
                 float *output){

    unsigned int channel_out = channel1 + channel2;
    for (int r = 0; r < height; ++r)
        for (int c = 0; c < width; ++c){
            std::memcpy(output + channel_out * width * r + channel_out * c, \
                   input1 + channel1 * width * r + channel1 * c, sizeof(float) * channel1);

            std::memcpy(output + channel_out * width * r + channel_out * c + channel1, \
                   input2 + channel2 * width * r + channel2 * c, sizeof(float) * channel2);
        }
}