#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <hip/hip_runtime.h>

#include "conv.h"
#include "Image.h"
#include "masks.h"

namespace fs = std::filesystem;

struct FilterChain {
    std::vector<MaskType> chain;
};

int main() {
    FilterChain filter_chain = {
        {SHARPEN, GAUSSIAN_BLUR_5x5, EDGE_DETECT, GAUSSIAN_BLUR_9x9}
    };

    std::ofstream csv("output/performance_results.csv");
    csv << "Resolution,CPUTime,GPUTimeWithMem,GPUTimeNoMem\n";

    std::string input_path = "input/";
    std::string output_path = "output/";
    fs::create_directories(output_path);

    for (const auto& entry : fs::directory_iterator(input_path)) {
        if (entry.path().extension() != ".jpg" && entry.path().extension() != ".png" && entry.path().extension() != ".bmp") continue;

        std::cout << "Measuring: " << entry.path().filename() << std::endl;

        // CPU processing
        {
            Image img_input(entry.path().string());
            
            // support buffers
            Image buffer1(img_input);
            Image buffer2(img_input.width(), img_input.height(), img_input.channels());

            Image* src = &buffer1;
            Image* dst = &buffer2;

            auto s_cpu = std::chrono::high_resolution_clock::now();
            for (const auto& filter : filter_chain.chain) {
                int mask_width;
                std::vector<float> mask = getMask(filter, mask_width);

                ImageConvolutionCPU(src->host(), dst->host(), src->width(), src->height(), src->channels(), mask.data(), mask_width);
                
                std::swap(src, dst);
            }
            auto e_cpu = std::chrono::high_resolution_clock::now();

            double t_cpu = std::chrono::duration<double, std::milli>(e_cpu - s_cpu).count();
            csv << img_input.width() << "x" << img_input.height() << ",";
            csv << t_cpu << ",";
            
            src->savejpg(output_path + entry.path().stem().string() + "_chain_cpu.jpg");
        }

        // GPU processing
        {
            Image img_input(entry.path().string());
            
            // support buffers
            Image buffer1(img_input);
            Image buffer2(img_input.width(), img_input.height(), img_input.channels());

            Image* src = &buffer1;
            Image* dst = &buffer2;

            // reorder to planar
            src->reorder_pixel_planar();

            src->device();
            dst->device();

            auto s_gpu = std::chrono::high_resolution_clock::now();
            for (const auto& filter : filter_chain.chain) {
                int mask_width;
                std::vector<float> mask = getMask(filter, mask_width);
                setConvolutionKernel(mask.data(), mask_width);

                for (int ch = 0; ch < src->channels(); ch++) {
                    ImageConvolutionGPUConstTiled(src->device() + ch * src->width() * src->height(),
                                                        dst->device() + ch * dst->width() * dst->height(),
                                                        src->width(), src->height(), mask_width);
                }
                
                std::swap(src, dst);
            }
            // ensure all operations are done
            HIP_CHECK_RETURN(hipDeviceSynchronize());
            auto e_gpu = std::chrono::high_resolution_clock::now();

            double t_gpu = std::chrono::duration<double, std::milli>(e_gpu - s_gpu).count();
            csv << t_gpu << "\n";

            src->sync_host();
            
            // reorder back to interleaved
            src->reorder_pixel_interleaved();
            src->savejpg(output_path + entry.path().stem().string() + "_chain_gpu.jpg");
        }

    }

    csv.close();
    return 0;
}