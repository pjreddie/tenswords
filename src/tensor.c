#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "tensor.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

tensor tensor_make(const size_t n, const size_t *size)
{
    tensor t = {0};
    t.n = n;
    t.size = calloc(n, sizeof(size_t));
    size_t i;
    for(i = 0; i < n; ++i){
        t.size[i] = size[i];
    }
    size_t len = tensor_len(t);
    t.data = calloc(len, sizeof(float));
    return t;
}

tensor tensor_copy(tensor t)
{
    tensor c = tensor_make(t.n, t.size);
    size_t i = 0;
    size_t len = tensor_len(t);
    for(i = 0; i < len; ++i){
        c.data[i] = t.data[i];
    }
    return c;
}

tensor tensor_scale(tensor t, float s)
{
    tensor c = tensor_copy(t);
    size_t i = 0;
    size_t len = tensor_len(c);
    for(i = 0; i < len; ++i){
        c.data[i] *= s;
    }
    return c;
}

tensor tensor_vmake(const size_t n, ...)
{
    size_t *size = calloc(n, sizeof(size_t));
    va_list args;
    va_start(args, n);
    size_t i;
    for(i = 0; i < n; ++i){
        size[i] = va_arg(args, size_t);
    }
    tensor t  = tensor_make(n, size);
    free(size);
    return t;
}

tensor tensor_random(const float s, const size_t n, const size_t *size)
{
    tensor t = tensor_make(n, size);
    size_t len = tensor_len(t);
    size_t i;
    for(i = 0; i < len; ++i){
        t.data[i] = 2*s*((float)rand()/RAND_MAX) - s;
    }
    return t;
}

tensor tensor_get_(const tensor t, const size_t e)
{
    assert (t.n > 0 && e >= 0 && e < t.size[0]);
    tensor a = {0};
    a.n = t.n - 1;
    a.size = t.size + 1;
    size_t len = tensor_len(a);
    a.data = t.data + e*len;
    return a;
}

tensor tensor_get(const tensor t, const size_t e)
{
    assert (e >= 0 && e < t.size[0]);
    tensor a = {0};

    if(t.n == 1){
        a.n = 1;
        a.size = malloc(sizeof(size_t));
        a.size[0] = 1;
        a.data = malloc(sizeof(float));
        a.data[0] = t.data[e];
        return a;
    }

    a.n = t.n - 1;
    a.size = calloc(a.n, sizeof(size_t));
    size_t i;
    size_t len = 1;
    for(i = 0; i < a.n; ++i){
        a.size[i] = t.size[i+1];
        len *= a.size[i];
    }
    a.data = calloc(len, sizeof(float));
    memcpy(a.data, t.data + len*e, len*sizeof(float));
    return a;
}

size_t tensor_len(const tensor t)
{
    size_t i;
    size_t len = 1;
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

void tensor_print(tensor t)
{
    size_t i;
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
    size_t ln = MIN(a.n, b.n);
    size_t i;
    for(i = 0; i < ln; ++i){
        size_t sa = a.size[a.n - 1 - i];
        size_t sb = b.size[b.n - 1 - i];
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

    size_t n  = a.n;
    size_t ln = b.n;

    size_t *size = calloc(n, sizeof(size_t));
    size_t i;
    for(i = 0; i < ln; ++i){
        size_t sa = a.size[a.n - 1 - i];
        size_t sb = b.size[b.n - 1 - i];
        size[n - 1 - i] = MAX(sa, sb);
    }
    for(i = 0; i < n - ln; ++i){
        size[i] = a.size[i];
    }
    tensor t = tensor_make(n, size);
    free(size);
    return t;
}

void tensor_binary_op_(const tensor a, const tensor b, tensor t, float op (float, float))
{
    if(t.n == 0){
        t.data[0] = op(a.data[0], b.data[0]);
    } else {
        size_t i = 0;
        for(i = 0; i < t.size[0]; ++i){
            size_t inca = (a.size[0] == t.size[0]);
            size_t incb = (b.size[0] == t.size[0]);
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

void tensor_axpy_(float a, tensor x, tensor y, tensor t)
{
    if(t.n == 0){
        t.data[0] = a * x.data[0] + y.data[0];
    } else if(t.n == 1){
        size_t i;
        size_t incx = (x.size[0] == t.size[0]);
        size_t incy = (y.size[0] == t.size[0]);
        for(i = 0; i < t.size[0]; ++i){
            t.data[i] = a*x.data[i*incx] + y.data[i*incy];
        }
    } else if (tensor_len(x) == tensor_len(t) && tensor_len(y) == tensor_len(t)) {
        size_t i;
        size_t len = tensor_len(t);
        for(i = 0; i < len; ++i){
            t.data[i] = a*x.data[i] + y.data[i];
        }
    } else {
        size_t i = 0;
        for(i = 0; i < t.size[0]; ++i){
            size_t incx = (x.size[0] == t.size[0]);
            size_t incy = (y.size[0] == t.size[0]);
            tensor subx = x;
            tensor suby = y;
            if(x.n == t.n){
                subx = tensor_get_(x, i*incx);
            }
            if(y.n == t.n){
                suby = tensor_get_(y, i*incy);
            }
            tensor subt = tensor_get_(t, i);
            tensor_axpy_(a, subx, suby, subt);
        }
    }
}

tensor tensor_binary_op(tensor a, tensor b, float op (float, float))
{
    tensor t = tensor_broadcast(a, b);
    if(t.data == 0) return t;
    tensor_binary_op_(a, b, t, op);
    return t;
}

tensor tensor_axpy(float a, tensor x, tensor y)
{
    tensor t = tensor_broadcast(x, y);
    if (t.data == 0) return t;
    tensor_axpy_(a, x, y, t);
    return t;
}

float tensor_add_op_(float a, float b)
{
    return a + b;
}

tensor tensor_add(tensor a, tensor b)
{
    return tensor_binary_op(a, b, tensor_add_op_);
}

tensor tensor_sub(tensor a, tensor b)
{
    return tensor_axpy(-1, b, a);
}

float tensor_mul_op_(float a, float b)
{
    return a * b;
}

tensor tensor_mul(tensor a, tensor b)
{
    return tensor_binary_op(a, b, tensor_mul_op_);
}

