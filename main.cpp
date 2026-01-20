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

struct TestConfig {
    MaskType type;
    std::string name;
};

int main() {
    std::vector<TestConfig> filters = {
        {SHARPEN, "Sharpen"},
        {EDGE_DETECT, "EdgeDetect"},
        {GAUSSIAN_BLUR_5x5, "Gaussian5x5"},
        {GAUSSIAN_BLUR_9x9, "Gaussian9x9"}
    };

    std::ofstream csv("output/performance_results.csv");
    csv << "Resolution,Filter,CPUTime,GPUTimeWithMem,GPUTimeNoMem\n";

    std::string input_path = "input/";
    std::string output_path = "output/";
    fs::create_directories(output_path);

    for (const auto& entry : fs::directory_iterator(input_path)) {
        if (entry.path().extension() != ".jpg" && entry.path().extension() != ".png" && entry.path().extension() != ".bmp") continue;

        std::cout << "Measuring: " << entry.path().filename() << std::endl;

        for (const auto& filter : filters) {
            Image img(entry.path().string());
            int mask_width;
            std::vector<float> mask = getMask(filter.type, mask_width);
            setConvolutionKernel(mask.data(), mask_width);

            /////////// CPU ///////////
            Image out_cpu(img.width(), img.height(), img.channels());
            auto s_cpu = std::chrono::high_resolution_clock::now();
            ImageConvolutionCPU(img.host(), out_cpu.host(), img.width(), img.height(), img.channels(), mask.data(), mask_width);
            auto e_cpu = std::chrono::high_resolution_clock::now();
            double t_cpu = std::chrono::duration<double, std::milli>(e_cpu - s_cpu).count();
            /////////// CPU ///////////

            /////////// GPU with Memory ///////////
            Image out_gpu(img.width(), img.height(), img.channels());
            auto s_gpu_mem = std::chrono::high_resolution_clock::now();
            
            img.device(); // sync to device
            ImageConvolutionGPUConstTiledInterleaved(img.device(), out_gpu.device(), img.width(), img.height(), img.channels(), mask_width);
            HIP_CHECK_RETURN(hipDeviceSynchronize());
            out_gpu.sync_host(); // invalidate host copy
            out_gpu.host(); // sync_host is lazy
            
            auto e_gpu_mem = std::chrono::high_resolution_clock::now();
            double t_gpu_mem = std::chrono::duration<double, std::milli>(e_gpu_mem - s_gpu_mem).count();

            /////////// GPU No Memory (Kernel only) ///////////
            auto s_gpu_nomem = std::chrono::high_resolution_clock::now();
            ImageConvolutionGPUConstTiledInterleaved(img.device(), out_gpu.device(), img.width(), img.height(), img.channels(), mask_width);
            HIP_CHECK_RETURN(hipDeviceSynchronize());
            auto e_gpu_nomem = std::chrono::high_resolution_clock::now();
            double t_gpu_nomem = std::chrono::duration<double, std::milli>(e_gpu_nomem - s_gpu_nomem).count();

            std::string res = std::to_string(img.width()) + "x" + std::to_string(img.height());
            csv << res << "," << filter.name << "," << t_cpu << "," << t_gpu_mem << "," << t_gpu_nomem << "\n";

            std::string base_name = entry.path().stem().string() + "_" + filter.name;
            out_cpu.savejpg(output_path + base_name + "_cpu.jpg");
            out_gpu.savejpg(output_path + base_name + "_gpu.jpg");
            
            std::cout << "  - " << filter.name << " complete. Speedup (Kernel): " << (t_cpu / t_gpu_nomem) << "x" << std::endl;
        }
    }

    csv.close();
    return 0;
}