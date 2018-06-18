//
// Created by Alex on 3/7/2017.
//

#ifndef LZMP_H
#define LZMP_H

/* Global define -----------------------------------------------------------*/
#define LZMP_ARRAY_MAX_SIZE             48000

// Enable print sub-strings
#define LZMP_SUB_STRINGS_PRINT_ENABLE

// Test Speed function interface
#define LZMP_TEST_SPEED_PRINT_ENABLE
#define LZMP_TEST_SPEED_NUMOF_NODES     2

/* Global typedef ----------------------------------------------------------*/

typedef struct {
	char data[LZMP_ARRAY_MAX_SIZE]; /*!< Data Array              */
	int len;                        /*!< Data Length             */
} LZMP_Data_TypeDef;

/* Global Call back functions ----------------------------------------------*/

/* Global function prototypes ----------------------------------------------*/

void LZMP_Calculate(LZMP_Data_TypeDef *h_Array_In, int Nodes, int *h_LZ_Out);

void LZMP_Test_Speed(void);
void LZMP_Test_Dictionary(void);

#endif //LZMP_H
