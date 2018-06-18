/*****************************************************************************
 * Created by Alex on 15-Jun-18.
 * Module ver 1.0
 * NOTE: This module needs 
 *
*****************************************************************************/

#ifndef LW_H
#define LW_H

/* Global define -----------------------------------------------------------*/
#define LW_ARRAY_MAX_SIZE             48000

// Test Speed function interface
#define LW_TEST_SPEED_PRINT_ENABLE
#define LW_TEST_SPEED_NUMOF_NODES     2

/* Global typedef ----------------------------------------------------------*/

/* Global Call back functions ----------------------------------------------*/
/* Global function prototypes ----------------------------------------------*/
void LW_Calculate_Min2(float *pV, int *pPsy, int m_len, int *pU, int M_len, int *pIndex_Out);

#endif //LW_H
