// Include guards and C++ compatibility
#ifndef TENSOR_H
#define TENSOR_H
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif



typedef struct tensor {
    size_t n;
    size_t *size;
    float *data;
} tensor;

tensor tensor_make(const size_t n, const size_t *size);
tensor tensor_vmake(const size_t n, ...);
tensor tensor_random(const float s, const size_t n, const size_t *size);
void   tensor_free(tensor t);
tensor tensor_get(const tensor t, const size_t e);
size_t    tensor_len(const tensor t);

void tensor_print(tensor t);
tensor tensor_copy(tensor t);
tensor tensor_scale(tensor t, float s);
int tensor_broadcastable(tensor a, tensor b);
tensor tensor_add(tensor a, tensor b);
tensor tensor_mul(tensor a, tensor b);
tensor tensor_axpy(float a, tensor x, tensor y);


#ifdef __cplusplus
}
#endif
#endif
