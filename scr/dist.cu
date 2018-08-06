//
// Created by Alex on 3/7/2017.
//
#include "dist.h"

/* Private define ------------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* ---------------------------------------------------------------------------*/

/**
 *@brief  Calculate distance
 *@param
 *@retval None
 */
__host__ __device__ float DIST_Calc(int v1, int v2)
{
    //ToDo: Need to change to distance function using Z type
    return (float) (v1 + v2);
}