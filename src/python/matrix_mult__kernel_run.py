#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Kernel tuner testing, kernel run
    
Description: 
    Simple test to check if the CUDA kernel runs within the kernel tuner.
    
-------------------------------------------------------------------------------    
Created on Fri May 27 12:31:11 2022
@author: apalha
"""

import numpy
from kernel_tuner import run_kernel


# %% Input parameters
n_columns_A = numpy.uint32(128)  # number of columns of A and rows of B
n_rows_A = numpy.uint32(128)  # number of rows of C (and rows of A)
n_columns_B = numpy.uint32(16384)  # number of columns of C (and columns of B)
n_iterations = numpy.uint32(100)  # number of times to perform the timing operation for averaging


# %% Initialization
# Matrix sizes
n_rows_B = n_columns_A
n_rows_D = n_rows_A
n_columns_D = n_columns_B


# %% Generate arrays
A = 1000.0*numpy.random.rand(n_rows_A, n_columns_A) + 100.0
B = 1000.0*numpy.random.rand(n_rows_B, n_columns_B) + 100.0
D_output = numpy.zeros([n_rows_D, n_columns_D])


# %% Expected result
D = A @ B


# %% Setup kernel run
kernel_name = "matrix_matrix_multiplication_gpu"
kernel_source = "matrix_matrix_multiplication_gpu.cu" 

problem_size = (n_columns_D, n_rows_D)

arguments = [D_output, A, B, n_columns_A, n_columns_D, n_rows_D]

params = dict()
params["block_size_x"] = 2
params["block_size_y"] = 512


# %% Run kernel
print('\n   Running naive algorithm (double precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params)

error_D = numpy.abs(answer[0] - D).max()

print("Max error in D:" + str(error_D))

print("Done")


# %% Setup optimized kernel run                                                                                                                                 
kernel_name = "matrix_matrix_multiplication_gpu_optimized"
kernel_source = "matrix_matrix_multiplication_gpu_optimized.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_output, A, B, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 8
params["block_size_y"] = 8
params["TILE_SIZE"] = params["block_size_x"]

# %% Run kernel
print('\n   Running optimized algorithm (double precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params)

error_D = numpy.abs(answer[0] - D).max()

print("Max error in D:" + str(error_D))

print("Done")



# %% Setup optimized kernel run (single precision)
kernel_name = "matrix_matrix_multiplication_gpu_optimized_single"
kernel_source = "matrix_matrix_multiplication_gpu_optimized_single.cu"

problem_size = (n_columns_D, n_rows_D)

A_single = A.astype(numpy.float32)
B_single = B.astype(numpy.float32)
D_output_single = D_output.astype(numpy.float32)

arguments = [D_output_single, A_single, B_single, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 8
params["block_size_y"] = 8
params["TILE_SIZE"] = params["block_size_x"]

# %% Run kernel
print('\n   Running optimized algorithm (single precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params)

error_D = numpy.abs(answer[0] - A_single @ B_single).max()

print("Max error in D:" + str(error_D))

print("Done")


# %% Setup optimized kernel run 2 (double precision)
kernel_name = "matrix_matrix_multiplication_gpu_optimized_2"
kernel_source = "matrix_matrix_multiplication_gpu_optimized_2.cu"

problem_size = (n_columns_D, n_rows_D)

A_single = A.astype(numpy.float32)
B_single = B.astype(numpy.float32)
D_output_single = D_output.astype(numpy.float32)

arguments = [D_output, A, B, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 16
params["block_size_y"] = 4
params["TILE_SIZE"] = params["block_size_x"]
params["VECTOR_SIZE"] = 4
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

# %% Run kernel
print('\n   Running optimized algorithm 2 (double precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

error_D = numpy.abs(answer[0] - A @ B).max()

print("Max error in D:" + str(error_D))

print("Done")


# %% Setup optimized kernel run 2 (single precision)
kernel_name = "matrix_matrix_multiplication_gpu_optimized_2_single"
kernel_source = "matrix_matrix_multiplication_gpu_optimized_2_single.cu"

problem_size = (n_columns_D, n_rows_D)

A_single = A.astype(numpy.float32)
B_single = B.astype(numpy.float32)
D_output_single = D_output.astype(numpy.float32)

arguments = [D_output_single, A_single, B_single, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 16
params["block_size_y"] = 4
params["TILE_SIZE"] = params["block_size_x"]
params["VECTOR_SIZE"] = 4
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

# %% Run kernel
print('\n   Running optimized algorithm 2 (single precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

error_D = numpy.abs(answer[0] - A_single @ B_single).max()

print("Max error in D:" + str(error_D))

print("Done")



# %% Setup optimized kernel run 3 (single precision)
kernel_name = "matrix_matrix_multiplication_gpu_optimized_3_single"
kernel_source = "matrix_matrix_multiplication_gpu_optimized_3_single.cu"

problem_size = (n_columns_D, n_rows_D)

A_single = A.astype(numpy.float32)
B_single = B.astype(numpy.float32)
D_output_single = D_output.astype(numpy.float32)

arguments = [D_output_single, A_single, B_single, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 16
params["block_size_y"] = 4
params["TILE_SIZE"] = params["block_size_x"]
params["VECTOR_SIZE"] = 4
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

# %% Run kernel
print('\n   Running optimized algorithm 3 (single precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

error_D = numpy.abs(answer[0] - A_single @ B_single).max()

print("Max error in D:" + str(error_D))

print("Done")
