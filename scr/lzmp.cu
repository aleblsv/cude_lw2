//
// Created by Alex on 3/7/2017.
//

#include <lzmp.h>

#include <assert.h>
#include <string.h>
#include <stdio.h>

#include <time.h>
#include <stdlib.h>

#include <helper_cuda.h>
#include <helper_functions.h>


/* Private define ------------------------------------------------------------*/

#define LZMP_SHARED_MEM_MAX_SIZE       48000  // For Device 3.5
#define LZMP_THREADS_PER_BLOCK         1024   // For Device 3.5

/* Private typedef -----------------------------------------------------------*/

/* Private macro -------------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
static void _LZMP_Launch(LZMP_Data_TypeDef *h_Data_In, int Nodes, int *h_LZ_Vec_Out);
__host__ __device__ void LZMP_Print_SubString(char *ptr, int start, int stop);
/* Private variables ---------------------------------------------------------*/
/* ---------------------------------------------------------------------------*/

/**
 *@brief  Print Sub-String
 *@param  ptr: device pointer to data
 *@param  start:
 *@param  stop:
 *@retval None
 */
__host__ __device__ void LZMP_Print_SubString(char *ptr, int start, int stop)
{
	for	(int i = start; i < stop; i++)
	{
		printf("%c", ptr[i]);
	}
	printf("\n");
}

/**
 *@brief  GPU - kernel
 *        Calculating LZ-complexity of a single node in a given array,
 *        Each block works on a calculation of a single node in a given array.
 *        Carefully launch blocks, blocks = number of nodes in an array.
 *@param  data: device pointer to data array
 *@param  lz_val: device pointer where kernel will copy after calculation
 *        LZ-complexity of each node in array
 *@retval None
 */
__device__ void LZMP_Calculate_Node(char *data, int size, int *lz_val)
{
    // initialize Thread index within a block
    int threadId = threadIdx.x;
    __shared__ int m; // history length
    __shared__ int SM; // maximum steps
    int D = 0; // lz variable  (Dictionary)
    int p = LZMP_THREADS_PER_BLOCK;
    int n = size;

    int i;
    int k;
    int h;
    int z;
    int j;
    int index;

    m = 0;
    while (m < n)
    {
        SM = 0;
        if (threadId >= 0 && threadId < p)
        {
            index = m / p;
            for (int l = 0; l < index + 1; l++)
            {
                // create new index that depend from threads
                j = threadId + (l * p);
                if (j < m)
                {
                    i = 0;
                    k = j;
                    h = m - j;
                    // Let each Thread scan and compare characters
                    // in history buffer with characters in S string
                    while (data[m + i] == data[k])
                    {
                        i++;
                        k++;
                        h--;
                        if (h == 0 || (m + i) == n)
                        {
                            break;
                        }
                    }
                    // If history is over and S is not
                    if (h == 0 && ((m + i) < n))
                    {
                        z = m;
                        // Let each Thread continue to scan &
                        // compare characters in S string
                        while (data[z] == data[m + i])
                        {
                            i++;
                            z++;
                            if ((m + i) == n)
                            {
                                break;
                            }
                        }
                    }
                    if (i > 0)
                    {
                        atomicMax(&SM, i);
                    }
                }
            }
        }
        __syncthreads();
        // Only first thread check and copy to history buffer
        if (threadId == 0)
        {
#ifdef LZMP_SUB_STRINGS_PRINT_ENABLE
#warning : FOR TEST ONLY!!!
        	LZMP_Print_SubString(data, m, (m + SM + 1));
#endif
            m += SM + 1;
            D++;
        }
        __syncthreads();
    }
    if (threadId == 0)
    {
        *lz_val = D;
    }
}

/**
 *@brief  GPU - kernel
 *        Calculating LZ-complexity of a single node in a given array,
 *        Each block works on a calculation of a single node in a given array.
 *        Carefully launch blocks, blocks = number of nodes in an array.
 *@param  ptr: device pointer to images pure data array,
 *        This parameter can be a value of @ref img_data_t
 *@param  lzmp_value: device pointer where kernel will copy after calculation
 *        LZ-complexity of each node in array
 *@retval None
 */
__global__ void LZMP_Kernel(LZMP_Data_TypeDef *ptr, int *lzmp_value)
{
    // initialize Thread index within a block
    int threadId = threadIdx.x;
    // each block work on specific string
    int node = blockIdx.x;
    int n = ptr[node].len;
    // initialize shared memory array, it much faster than global
    __shared__ char S[LZMP_SHARED_MEM_MAX_SIZE];

    if (threadId == 0)
    {
        assert(n <= LZMP_SHARED_MEM_MAX_SIZE);
        // copy current string from global to shared memory
        memcpy(S, ptr[node].data, n);
    }
    __syncthreads();
    LZMP_Calculate_Node(S, n, &lzmp_value[node]);
}

/**
 *@brief  Calculating LZMP values of Data Nodes (Array),
 *		  This function allocate memory on device (GPU)
 *		  Copy data from CPU to GPU and launch the kernel.
 *		  After calculation copy data back and deallocate device memory
 *@param  h_Data_In: host pointer to Data
 *        This parameter can be a value of @ref img_data_t
 *@param  Nodes: number of Data Nodes
 *@param  h_LZ_Vec_Out: host pointer to LZ Array
 *@retval None
 */
static void _LZMP_Launch(LZMP_Data_TypeDef *h_Data_In, int Nodes, int *h_LZ_Vec_Out)
{
    LZMP_Data_TypeDef *d_Data_In;
    int *d_LZ_Vec_Out;

    // Allocate memory on  Device
    checkCudaErrors(cudaMalloc((void **) &d_Data_In, Nodes * sizeof(LZMP_Data_TypeDef)));
    checkCudaErrors(cudaMalloc((void **) &d_LZ_Vec_Out, Nodes * sizeof(int)));

    // Copy data strings from Host memory to Device memory
    checkCudaErrors(cudaMemcpy(d_Data_In, h_Data_In, Nodes * sizeof(LZMP_Data_TypeDef), cudaMemcpyHostToDevice));

    int dimGrid = Nodes;
    int dimBlock = LZMP_THREADS_PER_BLOCK;
    // launch kernel
    LZMP_Kernel <<< dimGrid, dimBlock >>> (d_Data_In, d_LZ_Vec_Out);
    cudaDeviceSynchronize();

    // Copy result from Device memory to Host memory
    checkCudaErrors(cudaMemcpy(h_LZ_Vec_Out, d_LZ_Vec_Out, Nodes * sizeof(int), cudaMemcpyDeviceToHost));

    // Free Device memory
    checkCudaErrors(cudaFree(d_Data_In));
    checkCudaErrors(cudaFree(d_LZ_Vec_Out));
}

/**
 *@brief  Enter Function
 *@param  h_Data_In: host pointer to data nodes,
 *        This parameter can be a value of @ref img_data_t
 *@param  Nodes: number of Data Nodes
 *@param  h_LZ_Out: host pointer to LZ Array
 *@retval None
 */
void LZMP_Calculate(LZMP_Data_TypeDef *h_Data_In, int Nodes, int *h_LZ_Out)
{
    StopWatchInterface *timer = NULL;

    printf("\nGPU kernel - Start\n");
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    sdkStartTimer(&timer);
    _LZMP_Launch(h_Data_In, Nodes, h_LZ_Out);
    sdkStopTimer(&timer);

    printf("GPU kernel - Complete, time:%fms\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
}


/**
 *@brief  Test Speed Function
 *@param  none
 *@retval None
 */
void LZMP_Test_Speed(void)
{
    LZMP_Data_TypeDef *pData_Arr;
    int *pLZ_Arr;
    int i;
    int j;

	pData_Arr = (LZMP_Data_TypeDef *) malloc(LZMP_TEST_SPEED_NUMOF_NODES * sizeof(LZMP_Data_TypeDef));
	pLZ_Arr = (int *) malloc(LZMP_TEST_SPEED_NUMOF_NODES * sizeof(int));

	printf("Start Test Speed\n");
	if((pData_Arr == NULL) || (pLZ_Arr == NULL))
	{
		printf("Can't Allocate Memory, return\n");
		return;
	}

    /* initialize random seed: */
    srand(time(NULL));

    for (i = 0; i < LZMP_TEST_SPEED_NUMOF_NODES; i++)
    {
        for (j = 0; j < LZMP_ARRAY_MAX_SIZE; j++)
        {
            pData_Arr[i].data[j] = (char) (rand() % 0xff);
        }
        pData_Arr[i].len = LZMP_ARRAY_MAX_SIZE;  // Can be changed
        pLZ_Arr[i] = 0;
    }

#ifdef LZMP_TEST_SPEED_PRINT_ENABLE
    for (i = 0; i < LZMP_TEST_SPEED_NUMOF_NODES; i++)
    {
        printf("Node %d, Length %d bytes, LZ = %d\n", i, pData_Arr[i].len, pLZ_Arr[i]);
    }
#endif

    LZMP_Calculate(pData_Arr, LZMP_TEST_SPEED_NUMOF_NODES, pLZ_Arr);

#ifdef LZMP_TEST_SPEED_PRINT_ENABLE
    for (i = 0; i < LZMP_TEST_SPEED_NUMOF_NODES; i++)
    {
        printf("Node %d, Length %d bytes, LZ = %d\n", i, pData_Arr[i].len, pLZ_Arr[i]);
    }
#endif
    free(pData_Arr);
    free(pLZ_Arr);
}


#define LZMP_TEST_STRING "ABCABCDABCDEFABCABCDABCDEF"

/**
 *@brief  Test Dictionary Function
 *@param  none
 *@retval None
 */
void LZMP_Test_Dictionary(void) {
	LZMP_Data_TypeDef *pData_Arr;
	int *pLZ_Arr;

	pData_Arr = (LZMP_Data_TypeDef *) malloc(sizeof(LZMP_Data_TypeDef));
	pLZ_Arr = (int *) malloc(sizeof(int));

	printf("Start Test Dictionary\n");
	if ((pData_Arr == NULL) || (pLZ_Arr == NULL)) {
		printf("Can't Allocate Memory, return\n");
		return;
	}

	strcpy(pData_Arr[0].data, LZMP_TEST_STRING);
	pData_Arr[0].len = strlen(LZMP_TEST_STRING);
	pLZ_Arr[0] = 0;

	printf("Input String: ");
	LZMP_Print_SubString(pData_Arr[0].data, 0, pData_Arr[0].len);
	printf("Length %d bytes, LZ = %d", pData_Arr[0].len, pLZ_Arr[0]);

	LZMP_Calculate(pData_Arr, 1, pLZ_Arr);

	printf("LZ = %d", pLZ_Arr[0]);

	free(pData_Arr);
	free(pLZ_Arr);
}
