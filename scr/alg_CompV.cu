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
    float tVal;

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
                        tVal = MAT_GetElement(V_Mat, row, i);
                        tVal++;
                        MAT_SetElement(V_Mat, row, i, tVal);
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
                      Tp_fMat_TypeDef V_Mat)
{
    StopWatchInterface *timer = NULL;

    Tp_fMat_TypeDef D_Mat;
    Tp_fVec_TypeDef d_dNUN_Vec;
    int *d_mutex;

    Tp_fMat_TypeDef h_S_Mat;
    size_t Size;
    MISC_Bl_Size_TypeDef DimBlck = MISC_Get_Block_Size();

    printf("\nGPU kernel CompV - Start\n");
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    d_Z_Row = Z_Vec;
    Size = d_Z_Row.Size * sizeof(Tp_Z_TypeDef);
    checkCudaErrors(cudaMalloc(&d_Z_Row.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_Z_Row.pElements, Z_Vec.pElements, Size, cudaMemcpyHostToDevice));

    d_Z_Col = Z_Vec;
    Size = d_Z_Col.Size * sizeof(Tp_Z_TypeDef);
    checkCudaErrors(cudaMalloc(&d_Z_Col.pElements, Size));
    checkCudaErrors(cudaMemcpy(d_Z_Col.pElements, Z_Vec.pElements, Size, cudaMemcpyHostToDevice));

    d_dNUN_Vec = dNUN_Vec;
    Size = d_dNUN_Vec.Size * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_dNUN_Vec.pElements, Size));
    checkCudaErrors(cudaMalloc(&d_mutex, sizeof(int) * d_dNUN_Vec.Size));

    d_S_Mat.Width = d_Z_Col.Size;
    d_S_Mat.Height = d_Z_Row.Size;
    Size = d_S_Mat.Width * d_S_Mat.Height * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_S_Mat.pElements, Size));
    h_S_Mat = d_S_Mat;
    h_S_Mat.pElements = (float *) malloc(Size);
    MAT_SetElementAll(h_S_Mat, 0.0);

    // Invoke kernel
    dim3 dimBlock(DimBlck.Bl_2d, DimBlck.Bl_2d);
    dim3 dimGrid((d_S_Mat.Width + dimBlock.x - 1) / dimBlock.x, (d_S_Mat.Height + dimBlock.y - 1) / dimBlock.y);
    ALG_CompV_Kernel << < dimGrid, dimBlock >> > (d_Z_Row, d_Z_Col, d_S_Mat);
    cudaDeviceSynchronize();

    int tdimBlock = DimBlck.Bl_1d;
    int tdimGrid = (int) ((d_S_Mat.Width + tdimBlock - 1) / tdimBlock);
    MAX_MIN_min_vec_2DMat_kernel << < tdimGrid, tdimBlock >> > (d_S_Mat, d_dNUN_Vec, d_mutex);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(h_S_Mat.pElements, d_S_Mat.pElements, Size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dNUN_Vec.pElements, d_dNUN_Vec.pElements, d_dNUN_Vec.Size * sizeof(float),
                               cudaMemcpyDeviceToHost));
    printf("S matrix:\n");
    MAT_PrintMat(h_S_Mat);
//
//    printf("S matrix print by vec:\n");
//    Tp_fVec_TypeDef tVec;
//    for (int i = 0; i < h_S_Mat.Height; ++i)
//    {
//        tVec.pElements = MAT_GetRow_Vec(h_S_Mat, i);
//        tVec.Size = h_S_Mat.Width;
//        MAT_PrintVecFloat(tVec);
//    }

    free(h_S_Mat.pElements);

//    Free device memory
    checkCudaErrors(cudaFree(d_Z_Row.pElements));
    checkCudaErrors(cudaFree(d_Z_Col.pElements));
    checkCudaErrors(cudaFree(d_S_Mat.pElements));
    checkCudaErrors(cudaFree(d_dNUN_Vec.pElements));
    checkCudaErrors(cudaFree(d_mutex));

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
    Tp_Z_TypeDef z_arr[] = {
            {TYPES_NUM_OF_FEATURES, {2.0, 3.0}, 1, 0},
            {TYPES_NUM_OF_FEATURES, {1.0, 5.0}, 1, 0},
            {TYPES_NUM_OF_FEATURES, {7.0, 3.0}, 3, 0}
    };
    float dNUN_arr[MISC_NUM_OF_ELEMENTS(z_arr)];
    Tp_Z_Vec_TypeDef Z_Vec;
    Tp_fVec_TypeDef dNUN;

    Z_Vec.Size = MISC_NUM_OF_ELEMENTS(z_arr);
    Z_Vec.pElements = z_arr;

    dNUN.Size = MISC_NUM_OF_ELEMENTS(z_arr);
    dNUN.pElements = dNUN_arr;

    for (int i = 0; i < Z_Vec.Size; i++)
    {
        printf("label:%d, is_proto:%d, num_of_features:%d ->[",
               Z_Vec.pElements[i].Label,
               Z_Vec.pElements[i].IsProto,
               Z_Vec.pElements[i].Size);
        for (int j = 0; j < Z_Vec.pElements[i].Size; j++)
        {
            printf("%.2f ", Z_Vec.pElements[i].Feature_Arr[j]);
        }
        printf("]\n");
    }
    ALG_CompV_Launch(Z_Vec, dNUN);

    printf("dNUN=");
    MAT_PrintVecFloat(dNUN);
}
