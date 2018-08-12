//
// Created by Alex on 11/8/2017.
//

#include <stdlib.h>
#include <stdio.h>

#include "alg_CompV.h"
#include "alg_compDistFromEx.h"
#include "alg_initDnun.h"

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

    // k = row
    // V_Mat already initialized to zero
    if (row < V_Mat.Height && col < V_Mat.Width)
    {
        if (r_Vec.pElements[col] == 0)
        {
            for (int i = 0; i < dNUN_Vec.Size; i++)
            {
                if (MAT_GetElement(D_Mat, i, col) < dNUN_Vec.pElements[i])
                {
                    if (Z_Vec.pElements[i].Label == (row + 1))
                    {
                        atomicAdd(MAT_GetElementRef(V_Mat, row, col), 1);
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
    Tp_Z_TypeDef z_arr[] = {
            {TYPES_NUM_OF_FEATURES, {2.0, 3.0}, 1, 0},
            {TYPES_NUM_OF_FEATURES, {1.0, 5.0}, 2, 0},
            {TYPES_NUM_OF_FEATURES, {7.0, 3.0}, 3, 0}
    };
    Tp_Z_TypeDef zl_arr[] = {
            {TYPES_NUM_OF_FEATURES, {3.0, 1.0}, 1, 1},
            {TYPES_NUM_OF_FEATURES, {2.0, 4.0}, 3, 1},
    };
    Tp_Z_TypeDef u_arr[] = {
            {TYPES_NUM_OF_FEATURES, {1.0, 2.0}, 0, 0},
            {TYPES_NUM_OF_FEATURES, {7.0, 3.0}, 0, 0},
    };
    float dNUN_arr[MISC_NUM_OF_ELEMENTS(z_arr)];
    int r_arr[MISC_NUM_OF_ELEMENTS(z_arr)] = {0};
    Tp_Z_Vec_TypeDef Z_Vec;
    Tp_Z_Vec_TypeDef Zl_Vec;
    Tp_Z_Vec_TypeDef U_Vec;
    Tp_fMat_TypeDef D_Mat;
    Tp_fMat_TypeDef V_Mat;
    Tp_fVec_TypeDef dNUN;
    Tp_intVec_TypeDef r_Vec;
    size_t Size;

    Z_Vec.Size = MISC_NUM_OF_ELEMENTS(z_arr);
    Z_Vec.pElements = z_arr;
    Zl_Vec.Size = MISC_NUM_OF_ELEMENTS(zl_arr);
    Zl_Vec.pElements = zl_arr;
    U_Vec.Size = MISC_NUM_OF_ELEMENTS(u_arr);
    U_Vec.pElements = u_arr;
    dNUN.Size = MISC_NUM_OF_ELEMENTS(z_arr);
    dNUN.pElements = dNUN_arr;
    r_Vec.Size = MISC_NUM_OF_ELEMENTS(z_arr);
    r_Vec.pElements = r_arr;

    D_Mat.Height = Z_Vec.Size;
    D_Mat.Width = U_Vec.Size;
    Size = D_Mat.Height * D_Mat.Width * sizeof(float);
    D_Mat.pElements = (float *) malloc(Size);
    if (D_Mat.pElements == NULL)
    {
        printf("Can't allocate memory\n");
        return;
    }
    V_Mat.Height = 3; //Number of labels
    V_Mat.Width = U_Vec.Size;
    Size = V_Mat.Height * V_Mat.Width * sizeof(float);
    V_Mat.pElements = (float *) malloc(Size);
    if (V_Mat.pElements == NULL)
    {
        printf("Can't allocate memory\n");
        return;
    }

    MAT_SetElementAll(D_Mat, 0.0);
    printf("Z vector\n");
    MISC_Print_Z_Vec(Z_Vec);
    printf("Zl vector\n");
    MISC_Print_Z_Vec(Zl_Vec);
    printf("U vector\n");
    MISC_Print_Z_Vec(U_Vec);

    ALG_compDistFromEx_Launch(Z_Vec, U_Vec, D_Mat);
    printf("D Mat\n");
    MAT_PrintMat(D_Mat);

    ALG_initDnun_Launch(Z_Vec, Zl_Vec, dNUN);
    printf("dNUN vector\n");
    MAT_PrintVecFloat(dNUN);

    ALG_CompV_Launch(D_Mat, dNUN, r_Vec, Z_Vec, V_Mat);
    printf("V Mat\n");
    MAT_PrintMat(V_Mat);

    free(D_Mat.pElements);
    free(V_Mat.pElements);
}
