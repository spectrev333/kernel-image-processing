#include <cstdio>
#include <hip/hip_runtime.h>
#include <iostream>

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

int main() {
    std::cout << "Hello, World!" << std::endl;

    const unsigned char image[] = {
        1, 5, 1, 5, 1,
        5, 1, 5, 1, 5,
        1, 5, 1, 5, 1,
        5, 1, 5, 1, 5,
        1, 5, 1, 5, 1,
    };

    unsigned char output_image[25];

    const float mask[] = {
        0.5, 0.5, 0.5,
        0.5, 0.5, 0.5,
        0.5, 0.5, 0.5,
    };

    ImageConvolutionCPU(image, output_image, 5, 5, 1, mask, 3);
    
    for(int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            printf("%2d ", output_image[y*5 + x]);
        }
        printf("\n");
    }

    return 0;
}