#include "conv.h"

__constant__ float const_mask[MAX_MASK_SIZE];

extern "C" void setConvolutionKernel(float *h_Kernel, int mask_width) {
    HIP_CHECK_RETURN(hipMemcpyToSymbol(const_mask, h_Kernel, mask_width * mask_width * sizeof(float)));
}

__global__ void ImageConvolutionGPUConstKernel(
    const unsigned char *input,
    unsigned char *output_image, 
    int width, int height, int channels,
    int mask_width
) {
    
    int offset = mask_width / 2;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0;

        // mask columns
        for (int dy = -offset; dy <= offset; dy++) {
            // mask rows
            int pixel_y = y + dy;
            if (pixel_y < 0 || pixel_y >= height) continue;

            for (int dx = -offset; dx <= offset; dx++) {
                int pixel_x = x + dx;
                if (pixel_x < 0 || pixel_x >= width) continue;

                sum += input[(pixel_y * width + pixel_x) * channels + c] * const_mask[(dy + offset) * mask_width + (dx+offset)];
            }
        }

        if (sum > 255.0) sum = 255.0;
        if (sum < 0.0)   sum = 0.0;

        output_image[(y * width + x) * channels + c] = (unsigned char)sum;
    }
}

extern "C" void ImageConvolutionGPUConst(
    const unsigned char *device_input,
    unsigned char *device_output, 
    int width, int height,
    int channels,
    int mask_width
) {
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    ImageConvolutionGPUConstKernel<<<dimGrid, dimBlock>>>(
        device_input, device_output, width, height, channels, mask_width
    );
}

__global__ void ImageConvolutionGPUConstTiledKernel(
    const unsigned char *input,
    unsigned char *output_image, 
    int width, int height,
    int mask_width
) {
    
    extern __shared__ unsigned char shared_mem[];
    int shared_mem_width = TILE_WIDTH + (mask_width - 1);
    int shared_mem_height = TILE_WIDTH + (mask_width - 1);
    int thread_linear_index = threadIdx.y * blockDim.x + threadIdx.x;

    int shared_mem_total = shared_mem_width * shared_mem_height;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int i = thread_linear_index; i < shared_mem_total; i += threads_per_block) {
        int local_x = i % shared_mem_width;
        int local_y = i / shared_mem_width;
        
        int global_x = blockIdx.x * TILE_WIDTH + local_x - (mask_width / 2);
        int global_y = blockIdx.y * TILE_WIDTH + local_y - (mask_width / 2);
        
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height)
            shared_mem[i] = input[global_y * width + global_x];
        else
            shared_mem[i] = 0;
    }

    __syncthreads();

    int offset = mask_width / 2;
    float sum = 0.0;

    int shared_x = threadIdx.x + offset;
    int shared_y = threadIdx.y + offset;

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    // mask columns
    for (int dy = -offset; dy <= offset; dy++) {
        // mask rows
        int pixel_y = shared_y + dy;
        if (pixel_y < 0 || pixel_y >= shared_mem_height) continue;

        for (int dx = -offset; dx <= offset; dx++) {
            int pixel_x = shared_x + dx;
            if (pixel_x < 0 || pixel_x >= shared_mem_width) continue;

            sum += shared_mem[pixel_y * shared_mem_width + pixel_x] * const_mask[(dy + offset) * mask_width + (dx+offset)];
        }
    }

    if (sum > 255.0) sum = 255.0;
    if (sum < 0.0)   sum = 0.0;

    if (x < width && y < height) {
        output_image[y * width + x] = (unsigned char)sum;
    }
}

extern "C" void ImageConvolutionGPUConstTiled(
    const unsigned char *device_input,
    unsigned char *device_output, 
    int width, int height,
    int mask_width
) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);
    size_t shared_mem_size = (TILE_WIDTH + mask_width - 1) * (TILE_WIDTH + mask_width - 1) * sizeof(unsigned char);

    ImageConvolutionGPUConstTiledKernel<<<dimGrid, dimBlock, shared_mem_size>>>(
        device_input, device_output, width, height, mask_width
    );
}

__global__ void ImageConvolutionGPUConstTiledInterleavedKernel(
    const unsigned char *input,
    unsigned char *output_image, 
    int width, int height,
    int channels,
    int mask_width
) {
    
    extern __shared__ unsigned char shared_mem[];
    int shared_mem_width = TILE_WIDTH + (mask_width - 1);
    int shared_mem_height = TILE_WIDTH + (mask_width - 1);
    int thread_linear_index = threadIdx.y * blockDim.x + threadIdx.x;

    int shared_mem_total = shared_mem_width * shared_mem_height;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int i = thread_linear_index; i < shared_mem_total; i += threads_per_block) {
        int local_x = i % shared_mem_width;
        int local_y = i / shared_mem_width;
        
        int global_x = blockIdx.x * TILE_WIDTH + local_x - (mask_width / 2);
        int global_y = blockIdx.y * TILE_WIDTH + local_y - (mask_width / 2);
        
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height)
            for(int c = 0; c < channels; c++) {
                shared_mem[i * channels + c] = input[(global_y * width + global_x) * channels + c];
            }
        else
            for(int c = 0; c < channels; c++) {
                shared_mem[i * channels + c] = 0;
            }
    }

    __syncthreads();

    int offset = mask_width / 2;
    float sum = 0.0;

    int shared_x = threadIdx.x + offset;
    int shared_y = threadIdx.y + offset;

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

    for (int c = 0; c < channels; c++) {
        sum = 0.0;
        // mask columns
        for (int dy = -offset; dy <= offset; dy++) {
            // mask rows
            int pixel_y = shared_y + dy;
            if (pixel_y < 0 || pixel_y >= shared_mem_height) continue;

            for (int dx = -offset; dx <= offset; dx++) {
                int pixel_x = shared_x + dx;
                if (pixel_x < 0 || pixel_x >= shared_mem_width) continue;

                sum += shared_mem[(pixel_y * shared_mem_width + pixel_x) * channels + c] * const_mask[(dy + offset) * mask_width + (dx+offset)];
            }
        }

        if (sum > 255.0) sum = 255.0;
        if (sum < 0.0)   sum = 0.0;

        if (x < width && y < height) {
            output_image[(y * width + x) * channels + c] = (unsigned char)sum;
        }
    }
}

extern "C" void ImageConvolutionGPUConstTiledInterleaved(
    const unsigned char *device_input,
    unsigned char *device_output, 
    int width, int height,
    int channels,
    int mask_width
) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);
    size_t shared_mem_size = (TILE_WIDTH + mask_width - 1) * (TILE_WIDTH + mask_width - 1) * channels * sizeof(unsigned char);

    ImageConvolutionGPUConstTiledInterleavedKernel<<<dimGrid, dimBlock, shared_mem_size>>>(
        device_input, device_output, width, height, channels, mask_width
    );
}