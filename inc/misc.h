/*****************************************************************************
 * Created by Alex on 04-Aug-18.
 * Module ver 1.0
 * NOTE: This module needs 
*****************************************************************************/

#ifndef MISC_H
#define MISC_H

#include <stdint.h>

/* Global define ------------------------------------------------------------*/
#define MISC_NUM_OF_ELEMENTS(X)           (sizeof(X) / sizeof(X[0]))
/* Global typedef -----------------------------------------------------------*/
typedef struct
{
    int Bl_1d;
    int Bl_2d;
} MISC_Bl_Size_TypeDef;

/* Global Call back functions -----------------------------------------------*/
/* Global function prototypes -----------------------------------------------*/
void MISC_Check_Device(void);
MISC_Bl_Size_TypeDef MISC_Get_Block_Size(void);

#endif //MISC_H
