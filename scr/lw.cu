//
// Created by Alex on 15/6/2018.
//

#include <lw.h>

#include <assert.h>
#include <string.h>
#include <stdio.h>

#include <time.h>
#include <stdlib.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include "max_min.h"
#include "config.h"

/* Private define ------------------------------------------------------------*/

/* Private typedef -----------------------------------------------------------*/

/* Private macro -------------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
static void _LW_Launch_Min2(float *pV, int *pPsy, int m_len, int *pU, int M_len, int *pIndex_Out, float *pMin_Out);
/* Private variables ---------------------------------------------------------*/

/* ---------------------------------------------------------------------------*/

/**
 *@brief  GPU - kernel
 *@param
 *@retval None
 */
__global__ void
LW_Kernel_Min2(float *pMin_d, float *pV_d, int *pPsy_d, int m_len, int *pU_d, int M_len, int *pIndex_Out_d, int *mutex)
{
    int i_m = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    float minVal;

    if (i_m < m_len)
    {
        pMin_d[i_m] = MAX_MIN_INF;
        for (j = 0; j < M_len; j++)
        {
            if (pU_d[j] == 1)
            {
                if (pPsy_d[i_m] == j)
                {
                    pMin_d[i_m] = pV_d[i_m];
                    break;
                }
            }
        }
    }
    __syncthreads();

    find_minimum_index_kernel(pMin_d, &minVal, pIndex_Out_d, mutex, m_len);
}

/**
 *@brief  Pre-launch kernel function
 *@param
 *@retval None
 */
static void _LW_Launch_Min2(float *pV, int *pPsy, int m_len, int *pU, int M_len, int *pIndex_Out, float *pMin_Out)
{
    float *pMin_d;
    float *pV_d;
    int *pPsy_d;
    int *pU_d;
    int *pIndex_Out_d;
    int *d_mutex;

    // Allocate memory on  Device
    checkCudaErrors(cudaMalloc((void **) &pMin_d, m_len * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &pV_d, m_len * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &pPsy_d, m_len * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &pU_d, M_len * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &pIndex_Out_d, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_mutex, sizeof(int)));

    // Copy data from Host memory to Device memory
    checkCudaErrors(cudaMemcpy(pV_d, pV, m_len * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pPsy_d, pPsy, m_len * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pU_d, pU, M_len * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_mutex, 0, sizeof(int)));

    int threadsPerBlock = CONFIG_THREADS_PER_BLOCK;
    int blocksPerGrid = (m_len + threadsPerBlock - 1) / threadsPerBlock;
    // launch kernel
    LW_Kernel_Min2 << < blocksPerGrid, threadsPerBlock >> >
                                       (pMin_d, pV_d, pPsy_d, m_len, pU_d, M_len, pIndex_Out_d, d_mutex);
    cudaDeviceSynchronize();

    // Copy result from Device memory to Host memory
    checkCudaErrors(cudaMemcpy(pIndex_Out, pIndex_Out_d, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pMin_Out, pMin_d, m_len * sizeof(float), cudaMemcpyDeviceToHost));

    // Free Device memory
    checkCudaErrors(cudaFree(pMin_d));
    checkCudaErrors(cudaFree(pV_d));
    checkCudaErrors(cudaFree(pPsy_d));
    checkCudaErrors(cudaFree(pU_d));
    checkCudaErrors(cudaFree(pIndex_Out_d));
    checkCudaErrors(cudaFree(d_mutex));
}

/**
 *@brief  Enter Function
 *@param  pV: pointer to v array
 *@param  pPsy: pointer to psy array
 *@param  m_len: m length
 *@param  pU: pointer to u array
 *@param  M_len: M length
 *@param  pIndex_Out: Result, index of minimum value of v array
 *@retval None
 */
void LW_Calculate_Min2(float *pV, int *pPsy, int m_len, int *pU, int M_len, int *pIndex_Out)
{
    StopWatchInterface *timer = NULL;
    float ptMin[m_len];

    printf("\nGPU kernel - Start\n");
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    sdkStartTimer(&timer);
    _LW_Launch_Min2(pV, pPsy, m_len, pU, M_len, pIndex_Out, ptMin);
    sdkStopTimer(&timer);

    printf("GPU kernel - Complete, time:%fms\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
}

/**
 *@brief  Test
 *@param  none
 *@retval None
 */
void LW_Test_Min2(void)
{
    int index = 0;
    float v_arr[9] = {4, 52, 11, 5, 6, 7, 8, 9, 10};
    int psy_arr[9] = {0, 2, 9, 9, 9, 9, 9, 9, 0};
    int u_arr[3] = {1, 0, 1};

    printf("Start Test\n");
    LW_Calculate_Min2(v_arr, psy_arr, 9, u_arr, 3, &index);
    printf("Stop Test, v[%d]=%f\n", index, v_arr[index]);
}

