/*****************************************************************************
 * Created by Alex on 11-Aug-18.
 * Module ver 1.0
 * NOTE: This module needs 
*****************************************************************************/

#ifndef ALG_COMPV_H
#define ALG_COMPV_H

#include "config.h"
#include "types.h"

/* Global define ------------------------------------------------------------*/
/* Global typedef -----------------------------------------------------------*/
/* Global Call back functions -----------------------------------------------*/
/* Global function prototypes -----------------------------------------------*/
void ALG_CompV_Launch(const Tp_fMat_TypeDef D_Mat,
                      const Tp_fVec_TypeDef dNUN_Vec,
                      const Tp_intVec_TypeDef r_Vec,
                      const Tp_Z_Vec_TypeDef Z_Vec,
                      Tp_fMat_TypeDef V_Mat);
void ALG_CompV_Test(void);

#endif //ALG_COMPV_H
