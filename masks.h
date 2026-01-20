#include <vector>

typedef enum {
    RIDGE,
    EDGE_DETECT,
    SHARPEN,
    BOX_BLUR,
    GAUSSIAN_BLUR_3x3,
    GAUSSIAN_BLUR_5x5,
} MaskType;

std::vector<float> getMask(MaskType type, int &mask_width) {
    std::vector<float> mask;

    switch (type) {
        case RIDGE:
            mask_width = 3;
            mask = { 0, -1,  0, 
                    -1,  4, -1, 
                     0, -1,  0 };
            break;

        case EDGE_DETECT:
            mask_width = 3;
            mask = { -1, -1, -1, 
                     -1,  8, -1, 
                     -1, -1, -1 };
            break;

        case SHARPEN:
            mask_width = 3;
            mask = {  0, -1,  0, 
                     -1,  5, -1, 
                      0, -1,  0 };
            break;

        case BOX_BLUR:
            mask_width = 3;
            mask.assign(9, 1.0f / 9.0f);
            break;

        case GAUSSIAN_BLUR_3x3:
            mask_width = 3;
            mask = { 1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f,
                     2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f,
                     1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f };
            break;

        case GAUSSIAN_BLUR_5x5:
            mask_width = 5;
            mask = { 1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f,
                     4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
                     6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f, 6/256.0f,
                     4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
                     1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f };
            break;
    }

    return mask;
}