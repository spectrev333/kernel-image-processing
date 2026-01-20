#include <__clang_hip_runtime_wrapper.h>
#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define HIP_CHECK_RETURN(value) CheckHipErrorAux(__FILE__,__LINE__,#value,value)

static void CheckHipErrorAux(const char* file, unsigned line, const char* statement, hipError_t err) {
    if (err == hipSuccess) {
        return;
    }
    fprintf(stderr, "%s:%d: '%s' returned '%s' (%d)\n", file, line, statement, hipGetErrorString(err), err);
}

__global__ void HelloKernel() {
  printf("Hello from thread %d!\n", threadIdx.x);
}

void ImageConvolutionCPU(const unsigned char *input,
                         unsigned char *output_image, int width, int height,
                         int channels, const float *mask, int mask_width) {

    int offset = (mask_width) / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {                
                // kernel

                float sum = 0.0;

                // mask columns
                for (int dy = -offset; dy <= offset; dy++) {
                    // mask rows
                    int pixel_y = y + dy;
                    if (pixel_y < 0 || pixel_y >= height) continue;

                    for (int dx = -offset; dx <= offset; dx++) {
                        int pixel_x = x + dx;
                        if (pixel_x < 0 || pixel_x >= width) continue;

                        sum += input[(pixel_y * width + pixel_x) * channels + c] * mask[(dy + offset) * mask_width + (dx+offset)];
                        
                    }
                }

                if (sum > 255.0) sum = 255.0;
                if (sum < 0.0)   sum = 0.0;

                output_image[(y * width + x) * channels + c] = (unsigned char)sum;

            }
        }
    }
}

__global__ void ImageConvolutionGPU(
    const unsigned char *input,
    unsigned char *output_image, 
    int width, int height, int channels, 
    const float *mask, 
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


                sum += input[(pixel_y * width + pixel_x) * channels + c] * mask[(dy + offset) * mask_width + (dx+offset)];

            }
        }

        if (sum > 255.0) sum = 255.0;
        if (sum < 0.0)   sum = 0.0;

        output_image[(y * width + x) * channels + c] = (unsigned char)sum;
    }
}

#define MAX_MASK_SIZE 49

__constant__ float const_mask[MAX_MASK_SIZE];

__global__ void ImageConvolutionGPUConst(
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

#define TILE_WIDTH 16

// lets make some assumptiosns for simplicity
// TILE_WIDTH refers to the output tile width
// number of threads per block is assumed to be TILE_WIDTH x TILE_WIDTH
// mask width can be variable but should be odd

// based on previous assumptions, the shared memory size can be calculated as follows:
// shared memory width = TILE_WIDTH + (mask_width - 1)
// shared memory height = TILE_WIDTH + (mask_width - 1)

// to load data into shared memory, each thread will load one or more pixels from global memory
// linearizing thread indices allows to determine which pixels to load
// thread_linear_index = threadIdx.y * blockDim.x + threadIdx.x

// tile_x and tile_y refer to the top-left corner of the output tile in global image coordinates
// calculated as follows:
// tile_x = blockIdx.x * TILE_WIDTH
// tile_y = blockIdx.y * TILE_WIDTH

// src_x and src_y refer to the coordinates in global image space from which data is loaded into shared memory
// calculated as follows:
// src_x = tile_x + thread_linear_index % shared_mem_width
// src_y = tile_y + thread_linear_index / shared_mem_width
// or as 
// src_index = tile_y + (thread_linear_index / shared_mem_width) * width + tile_x + (thread_linear_index % shared_mem_width)

// then, each thread computes its output pixel using data from shared memory
// this uses the same convolution logic as before but accesses shared memory instead of global memory

// finally, the computed output pixel is written back to global memory
// at the appropriate location based on tile_x, tile_y, threadIdx.x, and threadIdx.y
// output_x = tile_x + threadIdx.x
// output_y = tile_y + threadIdx.y
// output_image[output_y * width + output_x] = computed_value

__global__ void ImageConvolutionGPUConstTiled(
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

__global__ void ImageConvolutionGPUConstTiledInterleaved(
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

// utility to reorder pixels from interleaved to planar
void reorder_pixel_planar(const unsigned char* image, unsigned char* output, int size, int channels) {
    for (int i = 0; i < size; i++) {
        for (int c = 0; c < channels; c++) {
            output[c * size + i] = image[i * channels + c];
        }
    }
}

void reorder_pixel_interleaved(const unsigned char* image, unsigned char* output, int size, int channels) {
    for (int i = 0; i < size; i++) {
        for (int c = 0; c < channels; c++) {
            output[i * channels + c] = image[c * size + i];
        }
    }
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    const unsigned char image[] = {
        1, 5, 1, 5, 1,
        5, 1, 5, 1, 5,
        1, 5, 1, 5, 1,
        5, 1, 5, 1, 5,
        1, 5, 1, 5, 1,
    };
      
    const unsigned char image3c[] = {
        1, 1, 1,    5, 5, 5,    1, 1, 1,    5, 5, 5,    1, 1, 1,
        5, 5, 5,    1, 1, 1,    5, 5, 5,    1, 1, 1,    5, 5, 5,
        1, 1, 1,    5, 5, 5,    1, 1, 1,    5, 5, 5,    1, 1, 1,
        5, 5, 5,    1, 1, 1,    5, 5, 5,    1, 1, 1,    5, 5, 5,
        1, 1, 1,    5, 5, 5,    1, 1, 1,    5, 5, 5,    1, 1, 1,
    };

    unsigned char output_image[25*3];

    const float box_blur[] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
    };

    const float sharpen[] = {
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0,
    };

    ImageConvolutionCPU(image3c, output_image, 5, 5, 3, box_blur, 3);
    
    for(int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            printf("(");
            for (int c = 0; c < 3; c++) {
                printf("%2d ", output_image[(y*5 + x)*3 + c]);
            }
            printf(") ");
        }
        printf("\n");
    }

    printf("Running on GPU...\n");

    HIP_CHECK_RETURN(hipMemcpyToSymbol(const_mask, sharpen, 3 * 3 * sizeof(float)));

    unsigned char* device_input;
    HIP_CHECK_RETURN(hipMalloc(&device_input, 5 * 5 * 3 * sizeof(unsigned char)));
    HIP_CHECK_RETURN(hipMemcpy(device_input, image3c, 5 * 5 * 3 * sizeof(unsigned char), hipMemcpyHostToDevice));
    
    unsigned char* device_output;
    HIP_CHECK_RETURN(hipMalloc(&device_output, 5 * 5 * 3 * sizeof(unsigned char)));

    int threadsPerBlock = 16;
    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 gridSize((5 + blockSize.x - 1) / blockSize.x, (5 + blockSize.y - 1) / blockSize.y);
    
    int shmem_size = (TILE_WIDTH + 3 - 1) * (TILE_WIDTH + 3 - 1) * sizeof(unsigned char) * 3;

    ImageConvolutionGPUConstTiledInterleaved<<<gridSize, blockSize, shmem_size>>>(device_input, device_output, 5, 5, 3, 3);

    HIP_CHECK_RETURN(hipDeviceSynchronize());

    unsigned char* output_data = (unsigned char*)malloc(5 * 5 * 3 * sizeof(unsigned char));
    HIP_CHECK_RETURN(hipMemcpy(output_data, device_output, 5 * 5 * 3 * sizeof(unsigned char), hipMemcpyDeviceToHost));

    for(int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            printf("(");
            for (int c = 0; c < 3; c++) {
                printf("%2d ", output_data[(y*5 + x)*3 + c]);
            }
            printf(") ");
        }
        printf("\n");
    }

    printf("Running tiled version on GPU...\n");

    unsigned char planar_image[25*3];

    reorder_pixel_planar(image3c, planar_image, 25, 3);

    HIP_CHECK_RETURN(hipMemcpy(device_input, planar_image, 5 * 5 * 3 * sizeof(unsigned char), hipMemcpyHostToDevice));

    int shared_mem_size = (TILE_WIDTH + 3 - 1) * (TILE_WIDTH + 3 - 1) * sizeof(unsigned char);

    ImageConvolutionGPUConstTiled<<<gridSize, blockSize, shared_mem_size>>>(device_input, device_output, 5, 5, 3);
    device_input += 5 * 5; // move to next channel
    device_output += 5 * 5;
    ImageConvolutionGPUConstTiled<<<gridSize, blockSize, shared_mem_size>>>(device_input, device_output, 5, 5, 3);
    device_input += 5 * 5; // move to next channel
    device_output += 5 * 5;
    ImageConvolutionGPUConstTiled<<<gridSize, blockSize, shared_mem_size>>>(device_input, device_output, 5, 5, 3);
    
    HIP_CHECK_RETURN(hipDeviceSynchronize());

    HIP_CHECK_RETURN(hipMemcpy(output_data, device_output - 2 * 5 * 5, 5 * 5 * 3 * sizeof(unsigned char), hipMemcpyDeviceToHost));
    reorder_pixel_interleaved(output_data, output_image, 25, 3);
    
    for(int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            printf("(");
            for (int c = 0; c < 3; c++) {
                printf("%2d ", output_image[(y*5 + x)*3 + c]);
            }
            printf(") ");
        }
        printf("\n");
    }

    int width, height, channels;
    unsigned char* img_data = stbi_load("input/sample.jpg", &width, &height, &channels, 3);
    if (!img_data) return -1;

    unsigned char* output_real_image = (unsigned char*)malloc(width * height * 3);

    printf("Image loaded: %dx%d, channels: %d\n", width, height, channels);
    
    ImageConvolutionCPU(img_data, output_real_image, width, height, 3, sharpen, 3);
    printf("Saving output image...\n");

    stbi_write_jpg("output/output_cpu.jpg", width, height, 3, output_real_image, 90);

    HIP_CHECK_RETURN(hipFree(device_input));
    HIP_CHECK_RETURN(hipFree(device_output));
    HIP_CHECK_RETURN(hipMalloc(&device_input, width * height * 3 * sizeof(unsigned char)));
    HIP_CHECK_RETURN(hipMalloc(&device_output, width * height * 3 * sizeof(unsigned char)));

    HIP_CHECK_RETURN(hipMemcpy(device_input, img_data, width * height * 3 * sizeof(unsigned char), hipMemcpyHostToDevice));
    
    dim3 blockSizeImage(threadsPerBlock, threadsPerBlock);
    dim3 gridSizeImage((width + blockSizeImage.x - 1) / blockSizeImage.x, (height + blockSizeImage.y - 1) / blockSizeImage.y);
    
    shared_mem_size = (TILE_WIDTH + 3 - 1) * (TILE_WIDTH + 3 - 1) * sizeof(unsigned char) * 3;

    ImageConvolutionGPUConstTiledInterleaved<<<gridSizeImage, blockSizeImage, shared_mem_size>>>(device_input, device_output, width, height, 3, 3);
    HIP_CHECK_RETURN(hipDeviceSynchronize());
    HIP_CHECK_RETURN(hipMemcpy(output_real_image, device_output, width * height * 3 * sizeof(unsigned char), hipMemcpyDeviceToHost));

    stbi_write_jpg("output/output_gpu.jpg", width, height, 3, output_real_image, 90);

    stbi_image_free(img_data);
    free(output_real_image);
    return 0;
}