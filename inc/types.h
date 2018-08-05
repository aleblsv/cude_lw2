/*****************************************************************************
 * Created by Alex on 04-Aug-18.
 * Module ver 1.0
 * NOTE: This module needs 
*****************************************************************************/

#ifndef TYPES_H
#define TYPES_H

#include "jetbrains_ide.h"

/* Global define ------------------------------------------------------------*/

/* Global typedef -----------------------------------------------------------*/
// Mat definition on Cuda device
//             |y1,x1; y1,x2; y1,x3|
// Mat(x, y):= |y2,x1; y2,x2; y2,x3|
//             |y3,x1; y3,x2; y3,x3|
typedef struct
{
    size_t Width;     // for col, x
    size_t Height;    // for row, y
    float *Elements;
} Tp_fMat_TypeDef;

typedef struct
{
    size_t Width;     // for col, x
    size_t Height;    // for row, y
    int *Elements;
} Tp_intMat_TypeDef;

typedef struct
{
    size_t Size;
    float *Elements;
} Tp_fVec_TypeDef;

typedef struct
{
    size_t Size;
    int *Elements;
} Tp_intVec_TypeDef;

/* Global Call back functions -----------------------------------------------*/
/* Global function prototypes -----------------------------------------------*/


#endif //TYPES_H
