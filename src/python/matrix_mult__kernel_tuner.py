#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Kernel tuner testing, kernel tuning
    
Description: 
    Tuning of element matrix matrix multiplication CUDA kernel for
    DG Acoustics.
    
-------------------------------------------------------------------------------    
Created on Fri May 27 12:31:11 2022
@author: apalha
"""

import numpy
from kernel_tuner import tune_kernel


# %% Input parameters
n_columns_A = numpy.uint32(4*128)  # number of columns of A and rows of B
n_rows_A = numpy.uint32(4*128)  # number of rows of C (and rows of A)
n_columns_B = numpy.uint32(16384)  # number of columns of C (and columns of B)
n_iterations = numpy.uint32(100)  # number of times to perform the timing operation for averaging


# %% Initialization
# Matrix sizes
n_rows_B = n_columns_A
n_rows_D = n_rows_A
n_columns_D = n_columns_B


# %% Generate arrays
A = numpy.random.rand(n_rows_A, n_columns_A)
B = numpy.random.rand(n_rows_B, n_columns_B)
D_output = numpy.zeros([n_rows_D, n_columns_D])

A_single = A.astype(numpy.float32)
B_single = B.astype(numpy.float32)
D_output_single = D_output.astype(numpy.float32)

# %% Expected result
D = A @ B
D_output_single = A_single @ B_single


# %% Setup kernel run (double precision)
kernel_name = "matrix_matrix_multiplication_gpu"
kernel_source = "matrix_matrix_multiplication_gpu.cu" 

problem_size = (n_columns_D, n_rows_D)

arguments = [D_output, A, B, n_columns_A, n_columns_D, n_rows_D]

tune_params = dict()
tune_params["block_size_x"] = [8, 16, 32]
tune_params["block_size_y"] = [32, 16, 8]


# %% Run kernel (single precision)
print("\n\n ------------------------------------------------------------")
print("\n\n Naive (double precision) \n")
results, env = tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, verbose=True)

#error_D = numpy.abs(answer[0] - D).sum()

#print("Print results:")
#for result in results:
#    print(result)
#
#print("\nPrint environment")
#print(env)

#print("Error in D:" + str(error_D))

print("Done")


# %% Setup kernel run (single  precision)                                                                                                              
kernel_name = "matrix_matrix_multiplication_gpu_single"
kernel_source = "matrix_matrix_multiplication_gpu_single.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_output_single, A_single, B_single, n_columns_A, n_columns_D, n_rows_D]

tune_params = dict()
tune_params["block_size_x"] = [8, 16, 32]
tune_params["block_size_y"] = [32, 16, 8]


# %% Run kernel (single precision)
print("\n\n ------------------------------------------------------------")
print("\n\n Naive (single precision) \n")
results, env = tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, verbose=True)

#error_D = numpy.abs(answer[0] - D).sum()                                                                                                             

#print("Print results:")
#for result in results:
#    print(result)
#
#print("\nPrint environment")
#print(env)

print("Done!")

# %% Setup kernel run (optimized, double  precision)
kernel_name = "matrix_matrix_multiplication_gpu_optimized"
kernel_source = "matrix_matrix_multiplication_gpu_optimized.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_output, A, B, n_columns_A, n_columns_B, n_rows_A]

tune_params = dict()
tune_params["TILE_SIZE"] = [4, 8, 16, 32]
tune_params["block_size_x"] = [4, 8, 16, 32]
tune_params["block_size_y"] = [4, 8, 16, 32]
restrict = ["block_size_x==block_size_y", "TILE_SIZE==block_size_x"]


# %% Run kernel (optimized, double precision)
print("\n\n ------------------------------------------------------------")
print("\n\n Optimized (double precision) \n")
results, env = tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, verbose=True, restrictions=restrict)

#error_D = numpy.abs(answer[0] - D).sum()

#print("Print results:")
#for result in results:
#    print(result)
#
#print("\nPrint environment")
#print(env)

print("Done!")

# %% Setup kernel run (optimized, single  precision)
kernel_name = "matrix_matrix_multiplication_gpu_optimized_single"
kernel_source = "matrix_matrix_multiplication_gpu_optimized_single.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_output_single, A_single, B_single, n_columns_A, n_columns_B, n_rows_A]

tune_params = dict()
tune_params["TILE_SIZE"] = [4, 8, 16, 32]
tune_params["block_size_x"] = [4, 8, 16, 32]
tune_params["block_size_y"] = [4, 8, 16, 32]
restrict = ["block_size_x==block_size_y", "TILE_SIZE==block_size_x"]

# %% Run kernel (optimized, single precision)
print("\n\n ------------------------------------------------------------")
print("\n\n Optimized (single precision) \n")
results, env = tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, verbose=True, restrictions=restrict)

#error_D = numpy.abs(answer[0] - D).sum()

#for result in results:
#    print(result)
#
#print("\nPrint environment")
#print(env)

print("Done!")


# %% Setup kernel run (optimized 2, double  precision)
kernel_name = "matrix_matrix_multiplication_gpu_optimized_2"
kernel_source = "matrix_matrix_multiplication_gpu_optimized_2.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_output, A, B, n_columns_A, n_columns_B, n_rows_A]

tune_params = dict()
tune_params["TILE_SIZE"] = [16]
tune_params["block_size_x"] = [16]
tune_params["block_size_y"] = [4]
# restrict = ["block_size_x==block_size_y", "TILE_SIZE==block_size_x"]
tune_params["VECTOR_SIZE"] = [4]
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]


# %% Run kernel (optimized 2, single precision)
print("\n\n ------------------------------------------------------------")
print("\n\n Optimized 2 (double precision) \n")
results, env = tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, verbose=True, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

#error_D = numpy.abs(answer[0] - D).sum()

#for result in results:
#    print(result)
#
#print("\nPrint environment")
#print(env)

print("Done!")


# %% Setup kernel run (optimized 2, single  precision)
kernel_name = "matrix_matrix_multiplication_gpu_optimized_2_single"
kernel_source = "matrix_matrix_multiplication_gpu_optimized_2_single.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_output_single, A_single, B_single, n_columns_A, n_columns_B, n_rows_A]

tune_params = dict()
tune_params["TILE_SIZE"] = [16]
tune_params["block_size_x"] = [16]
tune_params["block_size_y"] = [4]
# restrict = ["block_size_x==block_size_y", "TILE_SIZE==block_size_x"]
tune_params["VECTOR_SIZE"] = [4]
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]


# %% Run kernel (optimized 2, single precision)
print("\n\n ------------------------------------------------------------")
print("\n\n Optimized 2 (single precision) \n")
results, env = tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, verbose=True, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

#error_D = numpy.abs(answer[0] - D).sum()

#for result in results:
#    print(result)
#
#print("\nPrint environment")
#print(env)

print("Done!")




# %% Setup kernel run (optimized 3, single  precision)
kernel_name = "matrix_matrix_multiplication_gpu_optimized_3_single"
kernel_source = "matrix_matrix_multiplication_gpu_optimized_3_single.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_output_single, A_single, B_single, n_columns_A, n_columns_B, n_rows_A]

tune_params = dict()
tune_params["TILE_SIZE"] = [16]
tune_params["block_size_x"] = [16]
tune_params["block_size_y"] = [4]
# restrict = ["block_size_x==block_size_y", "TILE_SIZE==block_size_x"]
tune_params["VECTOR_SIZE"] = [4]
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]


# %% Run kernel (optimized 2, single precision)
print("\n\n ------------------------------------------------------------")
print("\n\n Optimized 3 (single precision) \n")
results, env = tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params, verbose=True, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

#error_D = numpy.abs(answer[0] - D).sum()

#for result in results:
#    print(result)
#
#print("\nPrint environment")
#print(env)

print("Done!")
