//
// Created by Alex on 3/7/2017.
//

#include <stdlib.h>
#include <stdio.h>

#include "alg_initDnun.h"
#include "max_min.h"
#include "mat.h"
#include "dist.h"

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
void ALG_initDnun_Launch(const Tp_Z_Vec_TypeDef Z_Row, const Tp_Z_Vec_TypeDef Z_Col, Tp_fVec_TypeDef *pdNUN_Vec)
{
    StopWatchInterface *timer = NULL;

    Tp_Z_Vec_TypeDef d_Z_Row;
    Tp_Z_Vec_TypeDef d_Z_Col;
    Tp_fMat_TypeDef d_S_Mat;


    size_t Size;
    MISC_Bl_Size_TypeDef DimBlck = MISC_Get_Block_Size();

    printf("\nGPU kernel compDistFromEx - Start\n");
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    d_Z_Vec = Z_Vec;
    Size = d_Z_Vec.Size * sizeof(int);
    checkCudaErrors(cudaMalloc(&d_Z_Vec.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_Z_Vec.pElements, Z_Vec.pElements, Size, cudaMemcpyHostToDevice));

    d_U_Vec = U_Vec;
    Size = d_U_Vec.Size * sizeof(int);
    checkCudaErrors(cudaMalloc(&d_U_Vec.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_U_Vec.pElements, U_Vec.pElements, Size, cudaMemcpyHostToDevice));

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
    checkCudaErrors(cudaFree(d_Z_Vec.pElements));
    checkCudaErrors(cudaFree(d_U_Vec.pElements));
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
void ALG_initDnun_Test(void)
{
    int z_arr[] = {1, 3, 4, 5, 7};
    int u_arr[] = {8, 9, 10};
    Tp_intVec_TypeDef h_Z_Vec;
    Tp_intVec_TypeDef h_U_Vec;
    Tp_fMat_TypeDef h_D_Mat;
    size_t Size;

    h_Z_Vec.pElements = z_arr;
    h_Z_Vec.Size = sizeof(z_arr) / sizeof(z_arr[0]);
    h_U_Vec.pElements = u_arr;
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