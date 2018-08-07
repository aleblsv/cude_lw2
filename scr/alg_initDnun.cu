//
// Created by Alex on 3/7/2017.
//

#include <stdlib.h>
#include <stdio.h>

#include "alg_initDnun.h"
#include "max_min.h"
#include "mat.h"
#include "dist.h"
#include "misc.h"

/* Private define ------------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* ---------------------------------------------------------------------------*/

/**
 *@brief  GPU - kernel,
 *@param
 *@retval None
 */
__global__ void ALG_initDnun_Kernel(Tp_Z_Vec_TypeDef Z_Row, Tp_Z_Vec_TypeDef Z_Col, Tp_fMat_TypeDef S_Mat)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Z_Row.Size && col < Z_Col.Size)
    {
        MAT_SetElement(S_Mat, row, col, MAX_MIN_INF);
        if(Z_Row.pElements[row].Label != Z_Col.pElements[col].Label)
        {
            MAT_SetElement(S_Mat, row, col, DIST_Calc_Feat(Z_Row.pElements[row], Z_Col.pElements[col]));
        }
    }
}

/**
 *@brief
 *@param
 *@retval None
 */
void ALG_initDnun_Launch(const Tp_Z_Vec_TypeDef Z_Vec, Tp_fVec_TypeDef *pdNUN_Vec)
{
    StopWatchInterface *timer = NULL;

    Tp_Z_Vec_TypeDef d_Z_Row;
    Tp_Z_Vec_TypeDef d_Z_Col;
    Tp_fMat_TypeDef d_S_Mat;
    Tp_fVec_TypeDef d_dNUN_Vec;
    size_t Size;
    MISC_Bl_Size_TypeDef DimBlck = MISC_Get_Block_Size();

    printf("\nGPU kernel compDistFromEx - Start\n");
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    d_Z_Row = Z_Vec;
    Size = d_Z_Row.Size * sizeof(Tp_Z_TypeDef);
    checkCudaErrors(cudaMalloc(&d_Z_Row.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_Z_Row.pElements, Z_Vec.pElements, Size, cudaMemcpyHostToDevice));

    d_Z_Col = Z_Vec;
    Size = d_Z_Row.Size * sizeof(Tp_Z_TypeDef);
    checkCudaErrors(cudaMalloc(&d_Z_Row.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_Z_Row.pElements, Z_Vec.pElements, Size, cudaMemcpyHostToDevice));

    d_S_Mat.Width = d_Z_Col.Size;
    d_S_Mat.Height = d_Z_Row.Size;
    Size = d_S_Mat.Width * d_S_Mat.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_S_Mat.pElements, Size));

    // Invoke kernel
    dim3 dimBlock(DimBlck.Bl_2d, DimBlck.Bl_2d);
    dim3 dimGrid((d_S_Mat.Width + dimBlock.x - 1) / dimBlock.x, (d_S_Mat.Height + dimBlock.y - 1) / dimBlock.y);
    ALG_initDnun_Kernel << < dimGrid, dimBlock >> > (d_Z_Row, d_Z_Col, d_S_Mat);
    cudaDeviceSynchronize();

    MAT_PrintMat(d_S_Mat);
    //ToDo:

    checkCudaErrors(cudaMemcpy(pdNUN_Vec, d_dNUN_Vec, Size, cudaMemcpyDeviceToHost));

//    Free device memory
    checkCudaErrors(cudaFree(d_Z_Row.pElements));
    checkCudaErrors(cudaFree(d_Z_Col.pElements));
    checkCudaErrors(cudaFree(d_S_Mat.pElements));
    checkCudaErrors(cudaFree(d_dNUN_Vec.pElements));

    sdkStopTimer(&timer);
    printf("GPU kernel - Complete, time:%fms\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
}

/**
 *@brief
 *@param
 *@retval None
 */
void ALG_initDnun_Test(void)
{
    float feat1_arr[] = {2.0, 3.0};
    float feat2_arr[] = {1.0, 5.0};
    Tp_Z_TypeDef z_arr[] = {
            {(sizeof(feat1_arr), feat1_arr, 1, 0)},
            {(sizeof(feat2_arr), feat2_arr, 2, 0)}
    };
    Tp_Z_Vec_TypeDef Z_Vec;

    Z_Vec.Size = sizeof(z_arr) / sizeof(z_arr[0]);
    Z_Vec.pElements = z_arr;

    if (h_D_Mat.pElements == NULL)
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
    free(h_D_Mat.pElements);
}