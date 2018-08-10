/*****************************************************************************
 * Created by Alex on 28-Jul-18.
 * Module ver 1.0
 * NOTE: This module needs 

*****************************************************************************/
#ifndef MAX_MIN_H
#define MAX_MIN_H

#include "config.h"
#include "jetbrains_ide.h"

/* Global define ------------------------------------------------------------*/
#define MAX_MIN_INF         0x7f800000

/* Global typedef -----------------------------------------------------------*/
/* Global Call back functions -----------------------------------------------*/
/* Global function prototypes -----------------------------------------------*/
__global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n);
__global__ void find_maximum_index_kernel(float *array, float *max, int *maxIndex, int *mutex, unsigned int n);
__global__ void find_minimum_kernel(float *array, float *min, int *mutex, unsigned int n);
__global__ void find_minimum_index_kernel(float *array, float *min, int *minIndex, int *mutex, unsigned int n);

__device__ void MAX_MIN_minimum_vector_of_2DMat(float *array, float *min, int *mutex, unsigned int n);
#endif //MAX_MIN_H
