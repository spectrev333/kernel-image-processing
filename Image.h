#ifndef IMAGE_H
#define IMAGE_H

#include <hip/hip_runtime.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

class Image {
private:
    int w, h, c;
    unsigned char* h_data;
    unsigned char* d_data;
    bool device_synced = false;

    void check_hip(hipError_t err) {
        if (err != hipSuccess) {
            std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl;
            exit(1);
        }
    }

public:
    Image(const std::string& filename) : w(0), h(0), c(3), h_data(nullptr), d_data(nullptr), device_synced(false) {
        h_data = stbi_load(filename.c_str(), &w, &h, &c, 3);
        if (!h_data) {
            std::cerr << "Errore: Impossibile caricare " << filename << std::endl;
        }
    }

    Image(int width, int height, int channels = 3, bool allocate_host = true) : w(width), h(height), c(channels), d_data(nullptr), device_synced(false) {
        if (allocate_host) {
            h_data = (unsigned char*)malloc(w * h * c);
        } else {
            h_data = nullptr;
        }
    }

    ~Image() {
        freehost();
        freedevice();
    }

    unsigned char* host() {
        if (!h_data) {
            h_data = (unsigned char*)malloc(size_bytes());
        }
        if (d_data && !device_synced) {
            check_hip(hipMemcpy(h_data, d_data, size_bytes(), hipMemcpyDeviceToHost));
            device_synced = true;
        }
        return h_data;
    }

    unsigned char* device() {
        if (!d_data) {
            check_hip(hipMalloc(&d_data, size_bytes()));
            if (h_data) {
                check_hip(hipMemcpy(d_data, h_data, size_bytes(), hipMemcpyHostToDevice));
            }
            device_synced = true;
        }
        return d_data;
    }

    void sync_host() {
        device_synced = false;
    }

    void freehost() {
        if (h_data) {
            free(h_data);
            h_data = nullptr;
        }
    }

    void freedevice() {
        if (d_data) {
            check_hip(hipFree(d_data));
            d_data = nullptr;
        }
    }

    void savejpg(const std::string& filename) {
        this->host();
        if (h_data) stbi_write_jpg(filename.c_str(), w, h, c, h_data, 90);
    }

    void savepng(const std::string& filename) {
        this->host();
        if (h_data) stbi_write_png(filename.c_str(), w, h, c, h_data, w * c);
    }

    size_t size_bytes() const { return w * h * c; }
    int width() const { return w; }
    int height() const { return h; }
    int channels() const { return c; }
};

#endif