#include <iostream>
#include <vector>
#include <chrono>
#include <hip/hip_runtime.h>

#include <iostream>
#include <string>
#include <hip/hip_runtime.h>

#include "conv.h"
#include "Image.h"
#include "masks.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Utilizzo: " << argv[0] << " <percorso_immagine>" << std::endl;
        return -1;
    }

    Image img(argv[1]);
    if (img.host() == nullptr) {
        return -1;
    }

    Image img_output_cpu(img.width(), img.height(), img.channels());
    // doesn't preallocate host memory for output on GPU
    Image img_output_gpu(img.width(), img.height(), img.channels(), false);

    // get mask using predefined type
    int mask_width;
    std::vector<float> mask = getMask(GAUSSIAN_BLUR_5x5, mask_width);

    // evauluate CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    ImageConvolutionCPU(img.host(), img_output_cpu.host(), img.width(), img.height(), img.channels(), mask.data(), mask_width);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    
    // put the mask in constant memory on the GPU
    setConvolutionKernel(mask.data(), mask_width);

    // allocate GPU memory and copy input image
    img.device(); // Assicuriamoci che i dati siano sulla GPU
    img_output_gpu.device(); // Allochiamo memoria sulla GPU per l'output

    // evaluate GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    
    ImageConvolutionGPUConstTiledInterleaved(img.device(), img_output_gpu.device(), img.width(), img.height(), img.channels(), mask_width);
    
    HIP_CHECK_RETURN(hipDeviceSynchronize()); // Attendiamo la fine per una misura precisa
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end_gpu - start_gpu;

    // invalidate host copy and sync back
    img_output_gpu.sync_host();

    // quick & dirty performance analysis
    double speedup = cpu_time.count() / gpu_time.count();
    std::cout << "\n--- Performance Analysis ---" << std::endl;
    std::cout << "Risoluzione: " << img.width() << "x" << img.height() << std::endl;
    std::cout << "Tempo CPU:   " << cpu_time.count() << " ms" << std::endl;
    std::cout << "Tempo GPU:   " << gpu_time.count() << " ms (Kernel Tiled)" << std::endl;
    std::cout << "Speedup:     " << speedup << "x" << std::endl;
    std::cout << "----------------------------\n" << std::endl;

    // save output images
    img_output_cpu.savejpg("output/output_cpu.jpg");
    img_output_gpu.savejpg("output/output_gpu.jpg");

    return 0;
}