/*****************************************************************************
 * Created by Alex on 04-Aug-18.
 * Module ver 1.0
 * NOTE: This module needs 
*****************************************************************************/

#ifndef MAT_H
#define MAT_H

#include "config.h"
#include "types.h"
#include "jetbrains_ide.h"

/* Global define ------------------------------------------------------------*/
/* Global typedef -----------------------------------------------------------*/
/* Global Call back functions -----------------------------------------------*/
/* Global function prototypes -----------------------------------------------*/
__host__ __device__ float MAT_GetElement(const Tp_fMat_TypeDef Mat, size_t row, size_t col);
__host__ __device__ void MAT_SetElement(Tp_fMat_TypeDef Mat, size_t row, size_t col, float value);
__host__ __device__ void MAT_SetElementAll(Tp_fMat_TypeDef Mat, float value);
__host__ __device__ float *MAT_GetElementRef(const Tp_fMat_TypeDef Mat, size_t row, size_t col);
__host__ __device__ float *MAT_GetRow_Vec(const Tp_fMat_TypeDef Mat, size_t row);
__host__ __device__ void MAT_PrintMat(Tp_fMat_TypeDef Mat);
__host__ __device__ void MAT_PrintVecInt(Tp_intVec_TypeDef Vec);
__host__ __device__ void MAT_PrintVecFloat(Tp_fVec_TypeDef Vec);

void MAT_Mult(const Tp_fMat_TypeDef A, const Tp_fMat_TypeDef B, Tp_fMat_TypeDef C);
void MAT_Sum(const Tp_fMat_TypeDef A, const Tp_fMat_TypeDef B, Tp_fMat_TypeDef C);

void MAT_Mult_Test(void);

#endif //MAT_H
