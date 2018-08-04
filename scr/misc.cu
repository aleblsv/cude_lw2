//
// Created by Alex on 3/7/2017.
//

#include "config.h"
#include "misc.h"

/* Private define ------------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
static MISC_Bl_Size_TypeDef sMISC_Block_Size = {256, 16};
/* ---------------------------------------------------------------------------*/

/**
 *@brief Check device compute capability
 *@param
 *@retval None
 */
void MISC_Check_Device(void)
{
    int devID;
    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr,
                "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error,
               __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major,
               deviceProp.minor);
    }

    // Use a larger block size for Fermi and above
    sMISC_Block_Size.Bl_2d = (deviceProp.major < 2) ? 16 : 32;
    sMISC_Block_Size.Bl_1d = sMISC_Block_Size.Bl_2d * sMISC_Block_Size.Bl_2d;
}

/**
 *@brief Check device compute capability
 *@param
 *@retval None
 */
MISC_Bl_Size_TypeDef MISC_Get_Block_Size(void)
{
    return sMISC_Block_Size;
}
