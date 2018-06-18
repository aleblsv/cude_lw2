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


/* Private define ------------------------------------------------------------*/

#define LW_SHARED_MEM_MAX_SIZE         48000  // For Device 3.5
#define LW_THREADS_PER_BLOCK           1024   // For Device 3.5

/* Private typedef -----------------------------------------------------------*/

/* Private macro -------------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private variables ---------------------------------------------------------*/

/* ---------------------------------------------------------------------------*/

/**
 *@brief  GPU - kernel
 *@param
 *@retval None
 */
__global__ void LW_Kernel_Min2(float *pMin_d, float *pV_d, int *pPsy_d, int m_len, int *pU_d, int M_len, int *pIndex_Out_d)
{
    int i_m = blockDim.x * blockIdx.x + threadIdx.x;
    int j;

    if (i_m < m_len)
    {
        pMin_d[i_m] = FLOAT_MAX;
        for (j = 0; j < M_len; j++)
        {
            if(pU_d[j] == 1)
            {
                if(pPsy_d[i_m] == j)
                {
                    pMin_d[i_m] = pV_d[i_m];
                    break;
                }
            }
        }
    }
    __syncthreads();

    // Calculate minimum in array
}

/**
 *@brief  Pre-launch kernel function
 *@param
 *@retval None
 */
static void _LW_Launch_Min2(float *pV, int *pPsy, int m_len, int *pU, int M_len, int *pIndex_Out)
{
    float *pMin_d;
    float *pV_d;
    int *pPsy_d;
    int *pU_d;
    int *pIndex_Out_d;

    // Allocate memory on  Device
    checkCudaErrors(cudaMalloc((void **) &pMin_d, m_len));
    checkCudaErrors(cudaMalloc((void **) &pV_d, m_len));
    checkCudaErrors(cudaMalloc((void **) &pPsy_d, m_len));
    checkCudaErrors(cudaMalloc((void **) &pU_d, M_len));
    checkCudaErrors(cudaMalloc((void **) &pIndex_Out_d, sizeof(int)));

    // Copy data from Host memory to Device memory
    checkCudaErrors(cudaMemcpy(pV_d, pV, m_len, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pPsy_d, pPsy, m_len, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pU_d, pU, M_len, cudaMemcpyHostToDevice));

    int threadsPerBlock = LW_THREADS_PER_BLOCK;
    int blocksPerGrid = (m_len + threadsPerBlock - 1) / threadsPerBlock;
    // launch kernel
    LW_Kernel_Min2 <<< blocksPerGrid, threadsPerBlock >>> (pMin_d, pV_d, pPsy_d, m_len, pU_d, M_len, pIndex_Out_d);
    cudaDeviceSynchronize();

    // Copy result from Device memory to Host memory
    checkCudaErrors(cudaMemcpy(pIndex_Out, pIndex_Out_d, sizeof(int), cudaMemcpyDeviceToHost));

    // Free Device memory
    checkCudaErrors(cudaFree(pMin_d));
    checkCudaErrors(cudaFree(pV_d));
    checkCudaErrors(cudaFree(pPsy_d));
    checkCudaErrors(cudaFree(pU_d));
    checkCudaErrors(cudaFree(pIndex_Out_d));
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

    printf("\nGPU kernel - Start\n");
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    sdkStartTimer(&timer);
    _LW_Launch_Min2(pV, pPsy, m_len, pU, M_len, pIndex_Out);
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
    float v_arr[] = {4, 52, 11, 5, 6, 7, 8, 9, 10};
    int psy_arr[] = {0, 2, 9, 9, 9, 9, 9, 9, 0};
    int u_arr[] =   {1, 0, 1};

    printf("Start Test\n");
    if((pData_Arr == NULL) || (pLZ_Arr == NULL))
    {
        printf("Can't Allocate Memory, return\n");
        return;
    }

    /* initialize random seed: */
    srand(time(NULL));


    LZMP_Calculate(pData_Arr, LZMP_TEST_SPEED_NUMOF_NODES, pLZ_Arr);

    printf("Stop Test\n");
    free(pData_Arr);
    free(pLZ_Arr);
}

