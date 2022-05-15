// Include guards and C++ compatibility
#ifndef TENSOR_H
#define TENSOR_H
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif



typedef struct tensor {
    int n;
    int *size;
    float *data;
} tensor;

tensor tensor_make(const int n, const int *size);
tensor tensor_vmake(const int n, ...);
tensor tensor_random(const float s, const int n, const int *size);
void   tensor_free(tensor t);
tensor tensor_get(const tensor t, const int e);
int    tensor_len(const tensor t);

tensor matrix_multiply(const tensor a, const tensor b);
tensor matrix_transpose(const tensor a);
tensor matrix_invert(tensor m);
tensor solve_system(tensor M, tensor b);
void tensor_print(tensor t);
tensor tensor_copy(tensor t);
tensor tensor_scale(tensor t, float s);
int tensor_broadcastable(tensor a, tensor b);
tensor tensor_add(tensor a, tensor b);
tensor tensor_mul(tensor a, tensor b);


#ifdef __cplusplus
}
#endif
#endif
