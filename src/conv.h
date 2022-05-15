// Include guards and C++ compatibility
#ifndef CONV_H
#define CONV_H
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif


tensor conv2d(tensor im, tensor filters, size_t stride, size_t pad);
tensor conv2d_slow(tensor im, tensor filters, size_t stride, size_t pad);


#ifdef __cplusplus
}
#endif
#endif
