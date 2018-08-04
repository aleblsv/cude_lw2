/*****************************************************************************
 * Created by Alex on 04-Aug-18.
 * Module ver 1.0
 * NOTE: This module needs 
*****************************************************************************/

#ifndef MAT_H
#define MAT_H

#include "types.h"

#include "jetbrains_ide.h"
/* Global define ------------------------------------------------------------*/
#define MAT_BLOCK_SIZE      1024

/* Global typedef -----------------------------------------------------------*/
/* Global Call back functions -----------------------------------------------*/
/* Global function prototypes -----------------------------------------------*/
__host__ __device__ float MAT_GetElement(const Tp_fMat_TypeDef Mat, size_t row, size_t col);
__host__ __device__ void MAT_SetElement(Tp_fMat_TypeDef Mat, size_t row, size_t col, float value);

void MAT_Mult(const Tp_fMat_TypeDef A, const Tp_fMat_TypeDef B, Tp_fMat_TypeDef C);
void MAT_Sum(const Tp_fMat_TypeDef A, const Tp_fMat_TypeDef B, Tp_fMat_TypeDef C);
void MAT_Print(Tp_fMat_TypeDef Mat);
void MAT_Check_Device(void);

void MAT_Mult_Test(void);

#endif //MAT_H
