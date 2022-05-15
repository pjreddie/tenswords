#include <assert.h>
#include <stdlib.h>
#include "tensor.h"
#include "matrix.h"

tensor im2col(tensor im, size_t size_y, size_t size_x, size_t stride, size_t pad)
{
    assert(im.n == 3);
    size_t i, j, k;

    size_t im_c = im.size[0];
    size_t im_h = im.size[1];
    size_t im_w = im.size[2];

    size_t res_h = (im_h + 2*pad - size_y)/stride + 1;
    size_t res_w = (im_w + 2*pad - size_x)/stride + 1;

    size_t rows = im_c*size_y*size_x;
    size_t cols = res_w * res_h;
    tensor col = tensor_vmake(2, rows, cols);

    // TODO: 5.1
    // Fill in the column matrix with patches from the image

    for (i = 0; i < rows; ++i) {
        size_t dx = i%size_x - pad;
        size_t dy = (i/size_x)%size_y - pad;
        //printf("%ld %ld\n", dy, dx);
        size_t ic = i / (size_y*size_x);
        for(j = 0; j < im_h + 2*pad - size_y + 1; j += stride){
            for(k = 0; k < im_w + 2*pad - size_x + 1; k += stride){
                float val = 0;
                size_t iw = k + dx;
                size_t ih = j + dy;
                if(ih >= 0 && ih < im_h && iw >= 0 && iw < im_w){
                    val = im.data[ic*im_w*im_h + ih*im_w + iw];
                }
                col.data[i*cols + (j/stride)*res_w + k/stride] = val;
            }
        }
    }
    return col;
}

tensor conv2d(tensor im, tensor filters, size_t stride, size_t pad)
{
    assert(filters.n == 4);
    assert(im.n == 3);
    assert(filters.size[1] == im.size[0]); // Filters and image have same # channels

    size_t f_c = filters.size[1];
    size_t f_h = filters.size[2];
    size_t f_w = filters.size[3];

    size_t im_c = im.size[0];
    size_t im_h = im.size[1];
    size_t im_w = im.size[2];

    size_t res_c = filters.size[0];
    size_t res_h = (im_h + 2*pad - f_h)/stride + 1;
    size_t res_w = (im_w + 2*pad - f_w)/stride + 1;
    tensor col = im2col(im, f_h, f_w, stride, pad);

    size_t temp_size[2];
    temp_size[0] = filters.size[0];
    temp_size[1] = filters.size[1]*filters.size[2]*filters.size[3];
    filters.n = 2;
    filters.size = temp_size;

    tensor res = matrix_multiply(filters, col);
    tensor_free(col);
    assert(res.size[0] == res_c);
    assert(res.size[1] == res_h*res_w);
    res.n = 3;
    free(res.size);
    res.size = calloc(3, sizeof(size_t));
    res.size[0] = res_c;
    res.size[1] = res_h;
    res.size[2] = res_w;
    return res;
}

tensor conv2d_slow(tensor im, tensor filters, size_t stride, size_t pad)
{
    assert(filters.n == 4);
    assert(im.n == 3);
    assert(filters.size[1] == im.size[0]); // Filters and image have same # channels
    size_t f_c = filters.size[1];
    size_t f_h = filters.size[2];
    size_t f_w = filters.size[3];

    size_t im_c = im.size[0];
    size_t im_h = im.size[1];
    size_t im_w = im.size[2];

    size_t res_c = filters.size[0];
    size_t res_h = (im_h + 2*pad - f_h)/stride + 1;
    size_t res_w = (im_w + 2*pad - f_w)/stride + 1;
    tensor res = tensor_vmake(3, res_c, res_h, res_w);
    size_t x, y, z;
    size_t dx, dy;
    size_t c;
    for(z = 0; z < res_c; ++z){
        for(c = 0; c < f_c; ++c){
            for(y = 0; y < res_h; ++y){
                for(x = 0; x < res_w; ++x){
                    float sum = 0;
                    size_t f_i = z*f_c*f_h*f_w + c*f_h*f_w;
                    for(dy = 0; dy < f_h; ++dy){
                        size_t im_y = y*stride - pad + dy;
                        for(dx = 0; dx < f_w; ++dx){
                            size_t im_x = x*stride - pad + dx;
                            size_t im_i = c*im_h*im_w + im_y*im_w + im_x;
                            float imv = 0;
                            if(im_x >= 0 && im_x < im_w && im_y >= 0 && im_y < im_h){
                                imv = im.data[im_i];
                            }
                            float fv = filters.data[f_i++];
                            sum += imv*fv;
                        }
                    }
                    res.data[z*res_h*res_w + y*res_w + x] += sum;
                }
            }
        }
    }
    return res;
}
