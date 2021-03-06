//
// Created by Alex on 3/7/2017.
//

#include <stdlib.h>
#include <stdio.h>

#include "alg_compDistFromEx.h"
#include "types.h"
#include "misc.h"
#include "mat.h"
#include "dist.h"

/* Private define ------------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* ---------------------------------------------------------------------------*/

/**
 *@brief  GPU - kernel, calculate distances of Z and U vectors building D matrix
 *@param
 *@retval None
 */
__global__ void ALG_compDistFromEx_Kernel(Tp_Z_Vec_TypeDef Z_Vec, Tp_Z_Vec_TypeDef U_Vec, Tp_fMat_TypeDef D_Mat)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < D_Mat.Height && col < D_Mat.Width)
    {
        MAT_SetElement(D_Mat, row, col, DIST_Calc_Feat(Z_Vec.pElements[row], U_Vec.pElements[col]));
    }
}

/**
 *@brief
 *@param
 *@retval None
 */
void ALG_compDistFromEx_Launch(const Tp_Z_Vec_TypeDef Z_Vec, const Tp_Z_Vec_TypeDef U_Vec, Tp_fMat_TypeDef D_Mat)
{
    StopWatchInterface *timer = NULL;

    Tp_Z_Vec_TypeDef d_Z_Vec;
    Tp_Z_Vec_TypeDef d_U_Vec;
    Tp_fMat_TypeDef d_D_Mat;
    size_t Size;
    MISC_Bl_Size_TypeDef DimBlck = MISC_Get_Block_Size();

    printf("\nGPU kernel compDistFromEx - Start\n");
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    d_Z_Vec = Z_Vec;
    Size = d_Z_Vec.Size * sizeof(Tp_Z_TypeDef);
    checkCudaErrors(cudaMalloc(&d_Z_Vec.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_Z_Vec.pElements, Z_Vec.pElements, Size, cudaMemcpyHostToDevice));

    d_U_Vec = U_Vec;
    Size = d_U_Vec.Size * sizeof(Tp_Z_TypeDef);
    checkCudaErrors(cudaMalloc(&d_U_Vec.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_U_Vec.pElements, U_Vec.pElements, Size, cudaMemcpyHostToDevice));

    d_D_Mat = D_Mat;
    Size = d_D_Mat.Width * d_D_Mat.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_D_Mat.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_D_Mat.pElements, D_Mat.pElements, Size, cudaMemcpyHostToDevice));

    // Invoke kernel
    dim3 dimBlock(DimBlck.Bl_2d, DimBlck.Bl_2d);
    dim3 dimGrid((d_D_Mat.Width + dimBlock.x - 1) / dimBlock.x, (d_D_Mat.Height + dimBlock.y - 1) / dimBlock.y);
    ALG_compDistFromEx_Kernel << < dimGrid, dimBlock >> > (d_Z_Vec, d_U_Vec, d_D_Mat);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(D_Mat.pElements, d_D_Mat.pElements, Size, cudaMemcpyDeviceToHost));

//    Free device memory
    checkCudaErrors(cudaFree(d_Z_Vec.pElements));
    checkCudaErrors(cudaFree(d_U_Vec.pElements));
    checkCudaErrors(cudaFree(d_D_Mat.pElements));

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
    Tp_Z_TypeDef z_arr[] = {
            {TYPES_NUM_OF_FEATURES, {2.0, 3.0}, 1, 0},
            {TYPES_NUM_OF_FEATURES, {1.0, 5.0}, 1, 0},
            {TYPES_NUM_OF_FEATURES, {7.0, 3.0}, 3, 0}
    };
    Tp_Z_TypeDef u_arr[] = {
            {TYPES_NUM_OF_FEATURES, {1.0, 2.0}, 0, 0},
            {TYPES_NUM_OF_FEATURES, {7.0, 3.0}, 0, 0},
    };

    Tp_Z_Vec_TypeDef Z_Vec;
    Tp_Z_Vec_TypeDef U_Vec;
    Tp_fMat_TypeDef D_Mat;
    size_t Size;

    Z_Vec.pElements = z_arr;
    Z_Vec.Size = sizeof(z_arr) / sizeof(z_arr[0]);
    U_Vec.pElements = u_arr;
    U_Vec.Size = sizeof(u_arr) / sizeof(u_arr[0]);

    D_Mat.Height = Z_Vec.Size;
    D_Mat.Width = U_Vec.Size;
    Size = D_Mat.Height * D_Mat.Width * sizeof(float);
    D_Mat.pElements = (float *) malloc(Size);
    if (D_Mat.pElements == NULL)
    {
        printf("Can't allocate memory\n");
        return;
    }
    MAT_SetElementAll(D_Mat, 0.0);
    MISC_Print_Z_Vec(Z_Vec);
    MISC_Print_Z_Vec(U_Vec);
    MAT_PrintMat(D_Mat);
    ALG_compDistFromEx_Launch(Z_Vec, U_Vec, D_Mat);
    MAT_PrintMat(D_Mat);
    free(D_Mat.pElements);
}