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

/**
 *@brief  Calculate distance of two points by features vector
 *@param
 *@retval None
 */
__host__ __device__ float DIST_Calc_Feat(Tp_Z_TypeDef v1, Tp_Z_TypeDef v2)
{
    float val = 0;

    if (v1.Size == v2.Size)
    {
        for (int i = 0; i < v1.Size; i++)
        {
            //val += pow((v1.pFeature[i] - v2.pFeature[i]), 2);
        	val = v1.pFeature[i] + v2.pFeature[i];
        }
//        return sqrt(val);
        return val;
    }
    return -1; // Indicate fault situation
}
