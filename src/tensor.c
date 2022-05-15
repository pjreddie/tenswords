#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "tensor.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

tensor tensor_make(const int n, const int *size)
{
    tensor t = {0};
    t.n = n;
    t.size = calloc(n, sizeof(int));
    int i;
    for(i = 0; i < n; ++i){
        t.size[i] = size[i];
    }
    int len = tensor_len(t);
    t.data = calloc(len, sizeof(float));
    return t;
}

tensor tensor_copy(tensor t)
{
    tensor c = tensor_make(t.n, t.size);
    int i = 0;
    int len = tensor_len(t);
    for(i = 0; i < len; ++i){
        c.data[i] = t.data[i];
    }
    return c;
}

tensor tensor_scale(tensor t, float s)
{
    tensor c = tensor_copy(t);
    int i = 0;
    int len = tensor_len(c);
    for(i = 0; i < len; ++i){
        c.data[i] *= s;
    }
    return c;
}

tensor tensor_vmake(const int n, ...)
{
    int *size = calloc(n, sizeof(int));
    va_list args;
    va_start(args, n);
    int i;
    for(i = 0; i < n; ++i){
        size[i] = va_arg(args, int);
    }
    tensor t  = tensor_make(n, size);
    free(size);
    return t;
}

tensor tensor_random(const float s, const int n, const int *size)
{
    tensor t = tensor_make(n, size);
    int len = tensor_len(t);
    int i;
    for(i = 0; i < len; ++i){
        t.data[i] = 2*s*((float)rand()/RAND_MAX) - s;
    }
    return t;
}

tensor tensor_get_(const tensor t, const int e)
{
    assert (t.n > 0 && e >= 0 && e < t.size[0]);
    tensor a = {0};
    a.n = t.n - 1;
    a.size = t.size + 1;
    int len = tensor_len(a);
    a.data = t.data + e*len;
    return a;
}

tensor tensor_get(const tensor t, const int e)
{
    assert (e >= 0 && e < t.size[0]);
    tensor a = {0};

    if(t.n == 1){
        a.n = 1;
        a.size = malloc(sizeof(int));
        a.size[0] = 1;
        a.data = malloc(sizeof(float));
        a.data[0] = t.data[e];
        return a;
    }

    a.n = t.n - 1;
    a.size = calloc(a.n, sizeof(int));
    int i;
    int len = 1;
    for(i = 0; i < a.n; ++i){
        a.size[i] = t.size[i+1];
        len *= a.size[i];
    }
    a.data = calloc(len, sizeof(float));
    memcpy(a.data, t.data + len*e, len*sizeof(float));
    return a;
}

int tensor_len(const tensor t)
{
    int i;
    int len = 1;
    for(i = 0; i < t.n; ++i){
        len *= t.size[i];
    }
    return len;
}

void tensor_free(tensor t)
{
    free(t.size);
    free(t.data);
}

tensor matrix_multiply(const tensor a, const tensor b)
{
    assert(a.n == 2);
    assert(b.n == 2);
    assert(a.size[1] == b.size[0]);
    int M = a.size[0];
    int K = a.size[1];
    int N = b.size[1];
    int size[2] = {M, N};
    tensor t = tensor_make(2, size);
    int i, j, k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            for(j = 0; j < N; ++j){
                t.data[i*N + j] += a.data[i*K + k]*b.data[k*N + j];
            }
        }
    }
    return t;
}

tensor matrix_transpose(tensor a)
{
    assert(a.n == 2);
    int size[2] = {a.size[1], a.size[0]};
    tensor t = tensor_make(2, size);

    int i, j;
    int rows = t.size[0];
    int cols = t.size[1];
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            t.data[i*cols + j] = a.data[j*rows + i];
        }
    }

    return t;
}

// Used for matrix inversion
tensor augment_matrix(tensor m)
{
    assert(m.n == 2);
    int rows = m.size[0];
    int cols = m.size[1];
    int i,j;
    tensor c = tensor_vmake(2, rows, cols*2);
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            c.data[i*cols*2 + j] = m.data[i*cols + j];
        }
    }
    for(j = 0; j < rows; ++j){
        c.data[j*cols*2 + j+cols] = 1;
    }
    return c;
}

// Invert matrix m
tensor matrix_invert(tensor m)
{
    int i, j, k;
    //print_matrix(m);
    assert(m.n == 2);
    assert(m.size[0] == m.size[1]);

    tensor c = augment_matrix(m);
    tensor none = {0};
    //print_matrix(c);
    float **cdata = calloc(c.size[0], sizeof(float *));
    for(i = 0; i < c.size[0]; ++i){
        cdata[i] = c.data + i*c.size[1];
    }

    for(k = 0; k < c.size[0]; ++k){
        float p = 0.;
        int index = -1;
        for(i = k; i < c.size[0]; ++i){
            float val = fabs(cdata[i][k]);
            if(val > p){
                p = val;
                index = i;
            }
        }
        if(index == -1){
            fprintf(stderr, "Can't do it, sorry!\n");
            tensor_free(c);
            return none;
        }

        float *swap = cdata[index];
        cdata[index] = cdata[k];
        cdata[k] = swap;

        float val = cdata[k][k];
        cdata[k][k] = 1;
        for(j = k+1; j < c.size[1]; ++j){
            cdata[k][j] /= val;
        }
        for(i = k+1; i < c.size[0]; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.size[1]; ++j){
                cdata[i][j] +=  s*cdata[k][j];
            }
        }
    }
    for(k = c.size[0]-1; k > 0; --k){
        for(i = 0; i < k; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.size[1]; ++j){
                cdata[i][j] += s*cdata[k][j];
            }
        }
    }
    //print_matrix(c);
    tensor inv = tensor_make(2, m.size);
    for(i = 0; i < m.size[0]; ++i){
        for(j = 0; j < m.size[1]; ++j){
            inv.data[i*m.size[1] + j] = cdata[i][j+m.size[1]];
        }
    }
    tensor_free(c);
    free(cdata);
    //print_matrix(inv);
    return inv;
}

tensor solve_system(tensor M, tensor b)
{
    tensor none = {0};
    tensor Mt = matrix_transpose(M);
    tensor MtM = matrix_multiply(Mt, M);
    tensor MtMinv = matrix_invert(MtM);
    if(!MtMinv.data) return none;
    tensor Mdag = matrix_multiply(MtMinv, Mt);
    tensor a = matrix_multiply(Mdag, b);
    tensor_free(Mt);
    tensor_free(MtM);
    tensor_free(MtMinv);
    tensor_free(Mdag);
    return a;
}

void tensor_print(tensor t)
{
    int i;
    if(t.n == 1){
        printf("[");
        for(i = 0; i < t.size[0]; ++i){
            printf("%6.3f ", t.data[i]);
        }
        printf("]\n");
    }
    else{
        for(i = 0; i < t.size[0]; ++i){
            tensor g = tensor_get(t, i);
            tensor_print(g);
            tensor_free(g);
        }
    }
}

int tensor_broadcastable(tensor a, tensor b)
{
    int ln = MIN(a.n, b.n);
    int i;
    for(i = 0; i < ln; ++i){
        int sa = a.size[a.n - 1 - i];
        int sb = b.size[b.n - 1 - i];
        if (sa != 1 && sb != 1 && sa != sb) return 0;
    }
    return 1;
}

tensor tensor_broadcast(tensor a, tensor b)
{
    if (!tensor_broadcastable(a, b)){
        fprintf(stderr, "Can't broadcast tensors\n");
        tensor none = {0};
        return none;
    }
    if(a.n < b.n){
        tensor swap = a;
        a = b;
        b = swap;
    }

    int n  = a.n;
    int ln = b.n;

    int *size = calloc(n, sizeof(int));
    int i;
    for(i = 0; i < ln; ++i){
        int sa = a.size[a.n - 1 - i];
        int sb = b.size[b.n - 1 - i];
        size[n - 1 - i] = MAX(sa, sb);
    }
    for(i = 0; i < n - ln; ++i){
        size[i] = a.size[i];
    }
    tensor t = tensor_make(n, size);
    free(size);
    return t;
}


void tensor_binary_op_(const tensor a, const tensor b, tensor t, void op (const tensor, const tensor, tensor))
{
    if(t.n == 0){
        op(a, b, t);
    } else {
        int i = 0;
        for(i = 0; i < t.size[0]; ++i){
            int inca = (a.size[0] == t.size[0]);
            int incb = (b.size[0] == t.size[0]);
            tensor suba = a;
            tensor subb = b;
            if(a.n == t.n){
                suba = tensor_get_(a, i*inca);
            }
            if(b.n == t.n){
                subb = tensor_get_(b, i*incb);
            }
            tensor subt = tensor_get_(t, i);
            tensor_binary_op_(suba, subb, subt, op);
        }
    }
}

void tensor_add_op_(tensor a, tensor b, tensor t)
{
    t.data[0] = a.data[0] + b.data[0];
}

tensor tensor_add(tensor a, tensor b)
{
    tensor t = tensor_broadcast(a, b);
    if (t.data == 0) return t;
    tensor_binary_op_(a, b, t, tensor_add_op_);
    return t;
}

void tensor_mul_op_(tensor a, tensor b, tensor t)
{
    t.data[0] = a.data[0] * b.data[0];
}

tensor tensor_mul(tensor a, tensor b)
{
    tensor t = tensor_broadcast(a, b);
    if (t.data == 0) return t;
    tensor_binary_op_(a, b, t, tensor_mul_op_);
    return t;
}
