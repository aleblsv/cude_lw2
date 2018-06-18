################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../scr/main.cpp 

CU_SRCS += \
../scr/lzmp.cu 

CU_DEPS += \
./scr/lzmp.d 

OBJS += \
./scr/lzmp.o \
./scr/main.o 

CPP_DEPS += \
./scr/main.d 


# Each subdirectory must supply rules for building sources it contributes
scr/%.o: ../scr/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I/home/alex/cuda-workspace/lzmp/inc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I/home/alex/cuda-workspace/lzmp/inc -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

scr/%.o: ../scr/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I/home/alex/cuda-workspace/lzmp/inc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I/home/alex/cuda-workspace/lzmp/inc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


