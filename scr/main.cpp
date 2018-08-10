/*
 * main.cpp
 *
 *  Created on: Jan 21, 2014
 *      Author: alex
 */
#include <stdio.h>

#include "alg_compDistFromEx.h"
#include "alg_initDnun.h"
#include "lw.h"
#include "mat.h"
#include "misc.h"
#include "max_min.h"

/**
 *@brief  Main function
 *@param  argc: total members from command line
 *@param  argv: pointer to an array of strings
 *@retval int
 */
int main(int argc, char *argv[])
{
	MISC_Check_Device();
    //LW_Test_Min2();
//    MAT_Mult_Test();
//    ALG_compDistFromEx_Test();
    ALG_initDnun_Test();
}
