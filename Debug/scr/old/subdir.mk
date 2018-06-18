################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../scr/old/dictionary.cpp 

OBJS += \
./scr/old/dictionary.o 

CPP_DEPS += \
./scr/old/dictionary.d 


# Each subdirectory must supply rules for building sources it contributes
scr/old/%.o: ../scr/old/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -I/usr/include/opencv -I/home/alex/cuda-workspace/opencv_test/inc -I/usr/local/cuda-5.5/samples/6_Advanced -I/usr/local/cuda/samples/common/inc -G -g -O0  -odir "scr/old" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc -I/usr/include/opencv -I/home/alex/cuda-workspace/opencv_test/inc -I/usr/local/cuda-5.5/samples/6_Advanced -I/usr/local/cuda/samples/common/inc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


