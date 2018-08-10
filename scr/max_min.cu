//
// Created by Alex on 15/6/2018.
//
// Example from: https://bitbucket.org/jsandham/algorithms_in_cuda
//
//

#include <types.h>
#include "config.h"
#include "max_min.h"
#include "misc.h"
#include "mat.h"

/* Private define ------------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
static __device__ void _MAX_MIN_min_vec_2DMat(float *array, float *min, int *mutex, unsigned int n);
/* Private variables ---------------------------------------------------------*/
/* ---------------------------------------------------------------------------*/


/**
 *@brief  Find maximum in array,  GPU - kernel
 *@param
 *@retval None
 */
__global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;

    __shared__ float cache[CONFIG_THREADS_PER_BLOCK_1D];


    float temp = -MAX_MIN_INF;
    while (index + offset < n)
    {
        temp = fmaxf(temp, array[index + offset]);

        offset += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();


    // reduction
    unsigned int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
        }

        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0);  //lock
        *max = fmaxf(*max, cache[0]);
        atomicExch(mutex, 0);  //unlock
    }
}


/**
 *@brief  Find maximum in array,  GPU - kernel
 *@param
 *@retval None
 */
__global__ void find_maximum_index_kernel(float *array, float *max, int *maxIndex, int *mutex, unsigned int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;
    __shared__ float cache[CONFIG_THREADS_PER_BLOCK_1D];
    __shared__ int indexCache[CONFIG_THREADS_PER_BLOCK_1D];

    float temp = -1.0;
    int tempIndex = 0;
    while (index + offset < n)
    {
        if (temp < array[index + offset])
        {
            temp = array[index + offset];
            tempIndex = index + offset;
        }
        offset += stride;
    }
    cache[threadIdx.x] = temp;
    indexCache[threadIdx.x] = tempIndex;
    __syncthreads();

    // reduction
    unsigned int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            if (cache[threadIdx.x] < cache[threadIdx.x + i])
            {
                cache[threadIdx.x] = cache[threadIdx.x + i];
                indexCache[threadIdx.x] = indexCache[threadIdx.x + i];
            }
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0);  //lock
        if (*max < cache[0])
        {
            *max = cache[0];
            *maxIndex = indexCache[0];
        }
        atomicExch(mutex, 0);  //unlock
    }
}

/**
 *@brief  Find minimum in array,  GPU - kernel
 *@param
 *@retval None
 */
__global__ void find_minimum_kernel(float *array, float *min, int *mutex, unsigned int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;

    __shared__ float cache[CONFIG_THREADS_PER_BLOCK_1D];


    float temp = MAX_MIN_INF;
    while (index + offset < n)
    {
        temp = fminf(temp, array[index + offset]);

        offset += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();


    // reduction
    unsigned int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            cache[threadIdx.x] = fminf(cache[threadIdx.x], cache[threadIdx.x + i]);
        }

        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0);  //lock
//        *min = fminf(*min, cache[0]);
        *min = cache[0];
        atomicExch(mutex, 0);  //unlock
    }
}

/**
 *@brief  Find minimum in array of 2d mamtrix, device function
 *@param
 *@retval None
 */
static __device__ void _MAX_MIN_min_vec_2DMat(float *array, float *min, int *mutex, unsigned int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;

    __shared__ float cache[CONFIG_THREADS_PER_BLOCK_2D];


    float temp = MAX_MIN_INF;
    while (index + offset < n)
    {
        temp = fminf(temp, array[index + offset]);

        offset += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();


    // reduction
    unsigned int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            cache[threadIdx.x] = fminf(cache[threadIdx.x], cache[threadIdx.x + i]);
        }

        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0);  //lock
        *min = cache[0];
        atomicExch(mutex, 0);  //unlock
    }
}

/**
 *@brief  Find minimum in array of 2d mamtrix, GPU kernel
 *@param
 *@retval None
 */
__global__ void MAX_MIN_min_vec_2DMat_kernel(Tp_fMat_TypeDef MatIn, Tp_fVec_TypeDef VecOut)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int mutex;

    if (row < MatIn.Height)
    {
        _MAX_MIN_min_vec_2DMat(MAT_GetRow_Vec(MatIn, row), &VecOut.pElements[row], &mutex, MatIn.Width);
    }
}

/**
 *@brief  Find maximum in array,  GPU - kernel
 *@param
 *@retval None
 */
__global__ void find_minimum_index_kernel(float *array, float *min, int *minIndex, int *mutex, unsigned int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;
    __shared__ float cache[CONFIG_THREADS_PER_BLOCK_1D];
    __shared__ int indexCache[CONFIG_THREADS_PER_BLOCK_1D];

    float temp = MAX_MIN_INF;
    int tempIndex = 0;
    while (index + offset < n)
    {
        if (temp > array[index + offset])
        {
            temp = array[index + offset];
            tempIndex = index + offset;
        }
        offset += stride;
    }
    cache[threadIdx.x] = temp;
    indexCache[threadIdx.x] = tempIndex;
    __syncthreads();

    // reduction
    unsigned int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            if (cache[threadIdx.x] > cache[threadIdx.x + i])
            {
                cache[threadIdx.x] = cache[threadIdx.x + i];
                indexCache[threadIdx.x] = indexCache[threadIdx.x + i];
            }
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0);  //lock
        *min = cache[0];
        *minIndex = indexCache[0];
        atomicExch(mutex, 0);  //unlock
    }
}
