//
// Created by Alex on 11/8/2017.
//

#include <stdlib.h>
#include <stdio.h>

#include "alg_CompV.h"
#include "mat.h"
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
__global__ void ALG_CompV_Kernel(const Tp_fMat_TypeDef D_Mat,
                                 const Tp_fVec_TypeDef dNUN_Vec,
                                 const Tp_intVec_TypeDef r_Vec,
                                 const Tp_Z_Vec_TypeDef Z_Vec,
                                 Tp_fMat_TypeDef V_Mat)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // V_Mat already initialized to zero
    if (row < V_Mat.Height && col < V_Mat.Width)
    {
        if (r_Vec.pElements[col] == 0)
        {
            for (int i = 0; i < dNUN_Vec.Size; i++)
            {
                if (MAT_GetElement(D_Mat, row, i) < dNUN_Vec.pElements[i])
                {
                    if (Z_Vec.pElements[i].Label == row)
                    {
                        atomicAdd(MAT_GetElementRef(V_Mat, row, i), 1);
                    }
                }
            }
        }
    }
}

/**
 *@brief
 *@param
 *@retval None
 */
void ALG_CompV_Launch(const Tp_fMat_TypeDef D_Mat,
                      const Tp_fVec_TypeDef dNUN_Vec,
                      const Tp_intVec_TypeDef r_Vec,
                      const Tp_Z_Vec_TypeDef Z_Vec,
                      Tp_fMat_TypeDef V_Mat)
{
    StopWatchInterface *timer = NULL;
	Tp_fMat_TypeDef d_D_Mat;
	Tp_fVec_TypeDef d_dNUN_Vec;
	Tp_intVec_TypeDef d_r_Vec;
	Tp_Z_Vec_TypeDef d_Z_Vec;
	Tp_fMat_TypeDef d_V_Mat;
    size_t Size;
    MISC_Bl_Size_TypeDef DimBlck = MISC_Get_Block_Size();

    printf("\nGPU kernel CompV - Start\n");
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    d_D_Mat = D_Mat;
    Size = d_D_Mat.Width * d_D_Mat.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_D_Mat.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_D_Mat.pElements, D_Mat.pElements, Size, cudaMemcpyHostToDevice));

    d_dNUN_Vec = dNUN_Vec;
    Size = d_dNUN_Vec.Size * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_dNUN_Vec.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_dNUN_Vec.pElements, d_dNUN_Vec.pElements, Size, cudaMemcpyHostToDevice));

    d_r_Vec = r_Vec;
    Size = d_r_Vec.Size * sizeof(int);
    checkCudaErrors(cudaMalloc(&d_r_Vec.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_r_Vec.pElements, r_Vec.pElements, Size, cudaMemcpyHostToDevice));

    d_Z_Vec = Z_Vec;
    Size = d_Z_Vec.Size * sizeof(Tp_Z_TypeDef);
    checkCudaErrors(cudaMalloc(&d_Z_Vec.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_Z_Vec.pElements, Z_Vec.pElements, Size, cudaMemcpyHostToDevice));

    d_V_Mat = V_Mat;
    Size = d_V_Mat.Width * d_V_Mat.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_V_Mat.pElements, Size));
    checkCudaErrors(cudaMemset(d_V_Mat.pElements, 0x00, Size));

    // Invoke kernel
    dim3 dimBlock(DimBlck.Bl_2d, DimBlck.Bl_2d);
    dim3 dimGrid((d_V_Mat.Width + dimBlock.x - 1) / dimBlock.x, (d_V_Mat.Height + dimBlock.y - 1) / dimBlock.y);
    ALG_CompV_Kernel << < dimGrid, dimBlock >> > (d_D_Mat, d_dNUN_Vec, d_r_Vec, d_Z_Vec, d_V_Mat);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(V_Mat.pElements, d_V_Mat.pElements, Size, cudaMemcpyDeviceToHost));

//    Free device memory
    checkCudaErrors(cudaFree(d_D_Mat.pElements));
    checkCudaErrors(cudaFree(d_dNUN_Vec.pElements));
    checkCudaErrors(cudaFree(d_r_Vec.pElements));
    checkCudaErrors(cudaFree(d_Z_Vec.pElements));
    checkCudaErrors(cudaFree(d_V_Mat.pElements));

    sdkStopTimer(&timer);
    printf("GPU kernel - Complete, time:%fms\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
}

/**
 *@brief
 *@param
 *@retval None
 */
void ALG_CompV_Test(void)
{
//    Tp_Z_TypeDef z_arr[] = {
//            {TYPES_NUM_OF_FEATURES, {2.0, 3.0}, 1, 0},
//            {TYPES_NUM_OF_FEATURES, {1.0, 5.0}, 1, 0},
//            {TYPES_NUM_OF_FEATURES, {7.0, 3.0}, 3, 0}
//    };
//    float dNUN_arr[MISC_NUM_OF_ELEMENTS(z_arr)] = {};
//    Tp_Z_Vec_TypeDef Z_Vec;
//    Tp_fVec_TypeDef dNUN;
//
//    Z_Vec.Size = MISC_NUM_OF_ELEMENTS(z_arr);
//    Z_Vec.pElements = z_arr;
//
//    dNUN.Size = MISC_NUM_OF_ELEMENTS(z_arr);
//    dNUN.pElements = dNUN_arr;
//
//    for (int i = 0; i < Z_Vec.Size; i++)
//    {
//        printf("label:%d, is_proto:%d, num_of_features:%d ->[",
//               Z_Vec.pElements[i].Label,
//               Z_Vec.pElements[i].IsProto,
//               Z_Vec.pElements[i].Size);
//        for (int j = 0; j < Z_Vec.pElements[i].Size; j++)
//        {
//            printf("%.2f ", Z_Vec.pElements[i].Feature_Arr[j]);
//        }
//        printf("]\n");
//    }
//    ALG_CompV_Launch(Z_Vec, dNUN);
//
//    printf("dNUN=");
//    MAT_PrintVecFloat(dNUN);
}
