//
// Created by Alex on 3/7/2017.
//

#include <stdlib.h>
#include <stdio.h>

#include "alg.h"
#include "config.h"
#include "types.h"
#include "misc.h"
#include "mat.h"

/* Private define ------------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* ---------------------------------------------------------------------------*/

/**
 *@brief  GPU - kernel, calculate distance
 *@param
 *@retval None
 */
__host__ __device__ float ALG_Dist(int v1, int v2)
{
    //ToDo: Need to change to distance function
    return (float) (v1 + v2);
}

/**
 *@brief  GPU - kernel, calculate distances of Z and U vectors building D matrix
 *@param
 *@retval None
 */
__global__ void ALG_compDistFromEx_Kernel(Tp_intVec_TypeDef Z_Vec, Tp_intVec_TypeDef U_Vec, Tp_fMat_TypeDef D_Mat)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < D_Mat.Height && col < D_Mat.Width)
    {
        MAT_SetElement(D_Mat, row, col, ALG_Dist(Z_Vec.Elements[row], U_Vec.Elements[col]));
    }
}

/**
 *@brief
 *@param
 *@retval None
 */
void ALG_compDistFromEx_Launch(const Tp_intVec_TypeDef Z_Vec, const Tp_intVec_TypeDef U_Vec, Tp_fMat_TypeDef D_Mat)
{
    StopWatchInterface *timer = NULL;

    Tp_intVec_TypeDef d_Z_Vec;
    Tp_intVec_TypeDef d_U_Vec;
    Tp_fMat_TypeDef d_D_Mat;
    size_t Size;
    MISC_Bl_Size_TypeDef DimBlck = MISC_Get_Block_Size();

    printf("\nGPU kernel compDistFromEx - Start\n");
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    d_Z_Vec = Z_Vec;
    Size = d_Z_Vec.Size * sizeof(int);
    checkCudaErrors(cudaMalloc(&d_Z_Vec.Elements, Size));
    checkCudaErrors(cudaMemcpy(d_Z_Vec.Elements, Z_Vec.Elements, Size, cudaMemcpyHostToDevice));

    d_U_Vec = U_Vec;
    Size = d_U_Vec.Size * sizeof(int);
    checkCudaErrors(cudaMalloc(&d_U_Vec.Elements, Size));
    checkCudaErrors(cudaMemcpy(d_U_Vec.Elements, U_Vec.Elements, Size, cudaMemcpyHostToDevice));

    d_D_Mat = D_Mat;
    Size = d_D_Mat.Width * d_D_Mat.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_D_Mat.Elements, Size));
    checkCudaErrors(cudaMemcpy(d_D_Mat.Elements, D_Mat.Elements, Size, cudaMemcpyHostToDevice));

    // Invoke kernel
    dim3 dimBlock(DimBlck.Bl_2d, DimBlck.Bl_2d);
    dim3 dimGrid((d_D_Mat.Width + dimBlock.x - 1) / dimBlock.x, (d_D_Mat.Height + dimBlock.y - 1) / dimBlock.y);
    ALG_compDistFromEx_Kernel << < dimGrid, dimBlock >> > (d_Z_Vec, d_U_Vec, d_D_Mat);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(D_Mat.Elements, d_D_Mat.Elements, Size, cudaMemcpyDeviceToHost));

//    Free device memory
    checkCudaErrors(cudaFree(d_Z_Vec.Elements));
    checkCudaErrors(cudaFree(d_U_Vec.Elements));
    checkCudaErrors(cudaFree(d_D_Mat.Elements));

    sdkStopTimer(&timer);
    printf("GPU kernel - Complete, time:%fms\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
}

/**
 *@brief
 *@param
 *@retval None
 */
void ALG_compDistFromEx_Test(void)
{
    int z_arr[] = {1, 3, 4, 5, 7};
    int u_arr[] = {8, 9, 10};
    Tp_intVec_TypeDef h_Z_Vec;
    Tp_intVec_TypeDef h_U_Vec;
    Tp_fMat_TypeDef h_D_Mat;
    size_t Size;

    h_Z_Vec.Elements = z_arr;
    h_Z_Vec.Size = sizeof(z_arr) / sizeof(z_arr[0]);
    h_U_Vec.Elements = u_arr;
    h_U_Vec.Size = sizeof(u_arr) / sizeof(u_arr[0]);

    h_D_Mat.Height = h_Z_Vec.Size;
    h_D_Mat.Width = h_U_Vec.Size;
    Size = h_D_Mat.Height * h_D_Mat.Width * sizeof(float);
    h_D_Mat.Elements = (float *) malloc(Size);
    if (h_D_Mat.Elements == NULL)
    {
        printf("Can't allocate memory\n");
        return;
    }
    MAT_SetElementAll(h_D_Mat, 0.0);
    MAT_PrintVec(h_Z_Vec);
    MAT_PrintVec(h_U_Vec);
    MAT_PrintMat(h_D_Mat);
    ALG_compDistFromEx_Launch(h_Z_Vec, h_U_Vec, h_D_Mat);
    MAT_PrintMat(h_D_Mat);
    free(h_D_Mat.Elements);
}