//
// Created by Alex on 3/7/2017.
//

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
        MAT_SetElement(D_Mat, row, col, (Z_Vec.Elements[row] + Z_Vec.Elements[col])); // ToDo: Change to distance between 2 points
    }
}

/**
 *@brief
 *@param
 *@retval None
 */
void ALG_compDistFromEx_Launch(const Tp_intVec_TypeDef Z_Vec, const Tp_intVec_TypeDef U_Vec, Tp_fMat_TypeDef D_Mat)
{
    Tp_intVec_TypeDef d_Z_Vec;
    Tp_intVec_TypeDef d_U_Vec;
    Tp_fMat_TypeDef d_D_Mat;
    size_t Size;
    MISC_Bl_Size_TypeDef DimBlck = MISC_Get_Block_Size();

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
}

/**
 *@brief
 *@param
 *@retval None
 */
void ALG_compDistFromEx_Test(void)
{
    
}