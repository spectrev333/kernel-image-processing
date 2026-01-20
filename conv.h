
#ifndef CONV_H
#define CONV_H

#include "hip/hip_runtime.h"
#include <stdio.h>

#define HIP_CHECK_RETURN(value) CheckHipErrorAux(__FILE__,__LINE__,#value,value)

static void CheckHipErrorAux(const char* file, unsigned line, const char* statement, hipError_t err) {
    if (err == hipSuccess) {
        return;
    }
    fprintf(stderr, "%s:%d: '%s' returned '%s' (%d)\n", file, line, statement, hipGetErrorString(err), err);
}

void ImageConvolutionCPU(const unsigned char *input,
                         unsigned char *output_image, int width, int height,
                         int channels, const float *mask, int mask_width);

#define MAX_MASK_SIZE 81

extern "C" void setConvolutionKernel(float *h_Kernel, int mask_width);

extern "C" void ImageConvolutionGPUConst(const unsigned char *device_input,
                                 unsigned char *device_output, int width,
                                 int height, int channels, int mask_width);


#define TILE_WIDTH 16

extern "C" void ImageConvolutionGPUConstTiled(const unsigned char *device_input,
                                       unsigned char *device_output, int width,
                                       int height,
                                       int mask_width);

extern "C" void ImageConvolutionGPUConstTiledInterleaved(const unsigned char *device_input,
                                               unsigned char *device_output, int width,
                                               int height, int channels,
                                               int mask_width);

#endif