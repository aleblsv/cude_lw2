/*****************************************************************************
 * Created by Alex on 28-Jul-18.
 * Module ver 1.0
 * NOTE: This module needs 
 *
 * Define GPIO with labels as follow:

*****************************************************************************/

#ifndef CONFIG_H
#define CONFIG_H

// includes CUDA
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>

/* Global define ------------------------------------------------------------*/

#define CONFIG_SHARED_MEM_MAX_SIZE         48000  // For Device 3.5
#define CONFIG_THREADS_PER_BLOCK_1D        1024
#define CONFIG_THREADS_PER_BLOCK_2D        32

/* Global typedef -----------------------------------------------------------*/
/* Global Call back functions -----------------------------------------------*/
/* Global function prototypes -----------------------------------------------*/


#endif //CONFIG_H
