/*****************************************************************************
 * Created by Alex on 05-Aug-18.
 * Module ver 1.0
 * NOTE: This module needs 
*****************************************************************************/

#ifndef _ALG_H
#define _ALG_H

#include "config.h"
#include "types.h"

/* Global define ------------------------------------------------------------*/
/* Global typedef -----------------------------------------------------------*/
/* Global Call back functions -----------------------------------------------*/
/* Global function prototypes -----------------------------------------------*/
__host__ __device__ float ALG_Dist(int v1, int v2);
void ALG_compDistFromEx_Launch(const Tp_intVec_TypeDef Z_Vec, const Tp_intVec_TypeDef U_Vec, Tp_fMat_TypeDef D_Mat);
void ALG_compDistFromEx_Test(void);

#endif //_ALG_H
