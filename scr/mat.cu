//
// Created by Alex on 3/7/2017.
//

#include <stdio.h>
#include "mat.h"
#include "misc.h"

/* Private define ------------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
static int Mat_Block_Size = 16;
/* ---------------------------------------------------------------------------*/

/**
 *@brief  Get a matrix element
 *@param
 *@retval None
 */
__host__ __device__ float MAT_GetElement(const Tp_fMat_TypeDef Mat, size_t row, size_t col)
{
    return Mat.pElements[row * Mat.Width + col];
}

/**
 *@brief  Set a matrix element
 *@param
 *@retval None
 */
__host__ __device__ void MAT_SetElement(Tp_fMat_TypeDef Mat, size_t row, size_t col, float value)
{
    Mat.pElements[row * Mat.Width + col] = value;
}

/**
 *@brief  Set all matrix elements with value
 *@param
 *@retval None
 */
__host__ __device__ void MAT_SetElementAll(Tp_fMat_TypeDef Mat, float value)
{
    for (size_t i = 0; i < Mat.Height; i++)
    {
        for (size_t j = 0; j < Mat.Width; j++)
        {
            MAT_SetElement(Mat, i, j, value);
        }
    }
}

/**
 *@brief  Get a row vector
 *@param
 *@retval None
 */
__host__ __device__ float *MAT_GetRow_Vec(const Tp_fMat_TypeDef Mat, size_t row)
{
    return &Mat.pElements[row * Mat.Width + 0];
}

/**
 *@brief Print matrix value
 *@param
 *@retval None
 */
__host__ __device__ void MAT_PrintMat(Tp_fMat_TypeDef Mat)
{
    for (size_t i = 0; i < Mat.Height; i++)
    {
        for (size_t j = 0; j < Mat.Width; j++)
        {
            printf("%.1f ", MAT_GetElement(Mat, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

/**
 *@brief Print vector values
 *@param
 *@retval None
 */
__host__ __device__ void MAT_PrintVec(Tp_intVec_TypeDef Vec)
{
    for (size_t i = 0; i < Vec.Size; i++)
    {
        printf("%d ", Vec.pElements[i]);
    }
    printf("\n");
}

/**
 *@brief  GPU - kernel
 *@param
 *@retval None
 */
__global__ void MAT_MulKernel(Tp_fMat_TypeDef A, Tp_fMat_TypeDef B, Tp_fMat_TypeDef C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0.0;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A.Height && col < B.Width)
    {
        for (int e = 0; e < A.Width; ++e)
        {
            Cvalue += (MAT_GetElement(A, row, col)) * (MAT_GetElement(B, row, col));
        }
    }
    MAT_SetElement(C, row, col, Cvalue);
}

/**
 *@brief  GPU - kernel
 *@param
 *@retval None
 */
__global__ void MAT_SumKernel(Tp_fMat_TypeDef A, Tp_fMat_TypeDef B, Tp_fMat_TypeDef C)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < C.Height && col < C.Width)
    {
        MAT_SetElement(C, row, col, (MAT_GetElement(A, row, col) + MAT_GetElement(B, row, col)));
    }
}

/**
 *@brief Matrices multiplication
 *@param
 *@retval None
 */
void MAT_Mult(const Tp_fMat_TypeDef A, const Tp_fMat_TypeDef B, Tp_fMat_TypeDef C)
{
    Tp_fMat_TypeDef d_A;
    Tp_fMat_TypeDef d_B;
    Tp_fMat_TypeDef d_C;
    size_t Size;

    d_A = A;
    Size = A.Width * A.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_A.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_A.pElements, A.pElements, Size, cudaMemcpyHostToDevice));

    d_B = B;
    Size = B.Width * B.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_B.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_B.pElements, B.pElements, Size, cudaMemcpyHostToDevice));

    d_C = C;
    Size = C.Width * C.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_C.pElements, Size));

    // Invoke kernel
    dim3 dimBlock(Mat_Block_Size, Mat_Block_Size);
    dim3 dimGrid((B.Width + dimBlock.x - 1) / dimBlock.x, (A.Height + dimBlock.y - 1) / dimBlock.y);
    MAT_MulKernel << < dimGrid, dimBlock >> > (d_A, d_B, d_C);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(C.pElements, d_C.pElements, Size, cudaMemcpyDeviceToHost));

//    Free device memory
    checkCudaErrors(cudaFree(d_A.pElements));
    checkCudaErrors(cudaFree(d_B.pElements));
    checkCudaErrors(cudaFree(d_C.pElements));
}


/**
 *@brief Matrices sum
 *@param
 *@retval None
 */
void MAT_Sum(const Tp_fMat_TypeDef A, const Tp_fMat_TypeDef B, Tp_fMat_TypeDef C)
{
    StopWatchInterface *timer = NULL;

    Tp_fMat_TypeDef d_A;
    Tp_fMat_TypeDef d_B;
    Tp_fMat_TypeDef d_C;
    size_t Size;

    printf("\nGPU kernel MAT_Sum - Start\n");
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    d_A = A;
    Size = A.Width * A.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_A.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_A.pElements, A.pElements, Size, cudaMemcpyHostToDevice));

    d_B = B;
    Size = B.Width * B.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_B.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_B.pElements, B.pElements, Size, cudaMemcpyHostToDevice));

    d_C = C;
    Size = C.Width * C.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_C.pElements, Size));

    // Invoke kernel
    dim3 dimBlock(Mat_Block_Size, Mat_Block_Size);
    dim3 dimGrid((C.Width + dimBlock.x - 1) / dimBlock.x, (C.Height + dimBlock.y - 1) / dimBlock.y);
    MAT_SumKernel << < dimGrid, dimBlock >> > (d_A, d_B, d_C);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(C.pElements, d_C.pElements, Size, cudaMemcpyDeviceToHost));

//    Free device memory
    checkCudaErrors(cudaFree(d_A.pElements));
    checkCudaErrors(cudaFree(d_B.pElements));
    checkCudaErrors(cudaFree(d_C.pElements));

    sdkStopTimer(&timer);
    printf("GPU kernel - Complete, time:%fms\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
}

#define MAT_TEST_WIDTH     4
#define MAT_TEST_HEIGHT    4
#define MAT_TEST_SIZE      (MAT_TEST_WIDTH * MAT_TEST_HEIGHT)

/**
 *@brief Test
 *@param
 *@retval None
 */
void MAT_Mult_Test(void)
{
    MISC_Bl_Size_TypeDef tBl_Size;
    float A_Arr[MAT_TEST_SIZE] = {0.0};
    float B_Arr[MAT_TEST_SIZE] = {0.0};
    float C_Arr[MAT_TEST_SIZE] = {0.0};
    Tp_fMat_TypeDef h_A;
    Tp_fMat_TypeDef h_B;
    Tp_fMat_TypeDef h_C;

    tBl_Size = MISC_Get_Block_Size();
    Mat_Block_Size = tBl_Size.Bl_2d;

    h_A.Width = MAT_TEST_WIDTH;
    h_A.Height = MAT_TEST_HEIGHT;
    h_A.pElements = A_Arr;

    h_B.Width = MAT_TEST_WIDTH;
    h_B.Height = MAT_TEST_HEIGHT;
    h_B.pElements = B_Arr;

    h_C.Width = MAT_TEST_WIDTH;
    h_C.Height = MAT_TEST_HEIGHT;
    h_C.pElements = C_Arr;

    MAT_SetElement(h_A, 0, 0, 1);
    MAT_SetElement(h_A, 1, 1, 2);
    MAT_SetElement(h_B, 0, 0, 3);
    MAT_SetElement(h_B, 1, 1, 4);

    MAT_PrintMat(h_A);
    MAT_PrintMat(h_B);
    MAT_Sum(h_A, h_B, h_C);
    MAT_PrintMat(h_C);
}