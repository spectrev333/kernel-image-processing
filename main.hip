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

#define TILE_DIMENSION 32

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

    unsigned char output_image[25];

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

    float* device_mask;
    HIP_CHECK_RETURN(hipMalloc(&device_mask, 3 * 3 * sizeof(float)));
    HIP_CHECK_RETURN(hipMemcpy(device_mask, box_blur, 3 * 3 * sizeof(float), hipMemcpyHostToDevice));
    
    unsigned char* device_input;
    HIP_CHECK_RETURN(hipMalloc(&device_input, 5 * 5 * 3 * sizeof(unsigned char)));
    HIP_CHECK_RETURN(hipMemcpy(device_input, image3c, 5 * 5 * 3 * sizeof(unsigned char), hipMemcpyHostToDevice));
    
    unsigned char* device_output;
    HIP_CHECK_RETURN(hipMalloc(&device_output, 5 * 5 * 3 * sizeof(unsigned char)));

    int threadsPerBlock = 16;
    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 gridSize((5 + blockSize.x - 1) / blockSize.x, (5 + blockSize.y - 1) / blockSize.y);
    
    ImageConvolutionGPU<<<gridSize, blockSize>>>(device_input, device_output, 5, 5, 3, device_mask, 3);

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

    // int width, height, channels;
    // unsigned char* img_data = stbi_load("input/sample.jpg", &width, &height, &channels, 3);
    // if (!img_data) return -1;

    // unsigned char* output_data = (unsigned char*)malloc(width * height * 3);

    // printf("Image loaded: %dx%d, channels: %d\n", width, height, channels);
    // ImageConvolutionCPU(img_data, output_data, width, height, 3, sharpen, 3);

    // printf("Saving output image...\n");

    // stbi_write_jpg("output/output_cpu.jpg", width, height, 3, output_data, 90);

    // stbi_image_free(img_data);
    // free(output_data);

    return 0;
}