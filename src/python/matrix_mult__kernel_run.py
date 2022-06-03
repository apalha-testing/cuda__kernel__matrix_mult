#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Kernel tuner testing: kernel run

Description:
    Simple test to check if the CUDA kernels run within the kernel tuner.
    Tests the different version of the matrix multiplication CUDA kernels
    that implement D = A @ B (using numpy notation).

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
# Double precision
A_double = numpy.random.rand(n_rows_A, n_columns_A)
B_double = numpy.random.rand(n_rows_B, n_columns_B)
D_double_kernel = numpy.zeros([n_rows_D, n_columns_D])

# Single precision
A_single = A_double.astype(numpy.float32)
B_single = B_double.astype(numpy.float32)
D_single_kernel = D_double_kernel.astype(numpy.float32)

# %% Expected result
D_double = A_double @ B_double
D_single = A_single @ B_single


# %% Kernel runs

# %% Naive matrix multiplication algorithm (double precision)
# Setup kernel
kernel_name = "matrix_mult_naive_double"
kernel_source = "../cu/matrix_mult_naive_double.cu" 

problem_size = (n_columns_D, n_rows_D)

arguments = [D_double_kernel, A_double, B_double, n_columns_A, n_columns_D, n_rows_D]

params = dict()
params["block_size_x"] = 32
params["block_size_y"] = 32

# Run kernel
print('\n---------------------------------------------------------------------')
print('Running naive matrix multiplication algorithm (double precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params)

error_D = numpy.abs(answer[0] - D_double).max()

print("   Max error in D:" + str(error_D))
print("Done")
print('---------------------------------------------------------------------\n')

# %% Naive matrix multiplication algorithm (single precision)
# Setup kernel
kernel_name = "matrix_mult_naive_single"
kernel_source = "../cu/matrix_mult_naive_single.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_single_kernel, A_single, B_single, n_columns_A, n_columns_D, n_rows_D]

params = dict()
params["block_size_x"] = 32
params["block_size_y"] = 32

# Run kernel
print('\n---------------------------------------------------------------------')
print('Running naive matrix multiplication algorithm (single precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params)

error_D = numpy.abs(answer[0] - D_single).max()

print("    Max error in D:" + str(error_D))
print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm (double precision)
# Setup kernel
kernel_name = "matrix_mult_tiling_double"
kernel_source = "../cu/matrix_mult_tiling_double.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_double_kernel, A_double, B_double, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 8
params["block_size_y"] = 8
params["TILE_SIZE"] = params["block_size_x"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm (double precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params)

error_D = numpy.abs(answer[0] - D_double).max()

print("    Max error in D:" + str(error_D))

print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm (single precision)
# Setup kernel
kernel_name = "matrix_mult_tiling_single"
kernel_source = "../cu/matrix_mult_tiling_single.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_single_kernel, A_single, B_single, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 8
params["block_size_y"] = 8
params["TILE_SIZE"] = params["block_size_x"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm (single precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params)

error_D = numpy.abs(answer[0] - D_single).max()

print("    Max error in D:" + str(error_D))

print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm opimization 1 (no bank conflict and computation optimization) (double precision)
# Setup kernel
kernel_name = "matrix_mult_tiling_optimization_1_double"
kernel_source = "../cu/matrix_mult_tiling_optimization_1_double.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_double_kernel, A_double, B_double, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 16
params["block_size_y"] = 4
params["TILE_SIZE"] = params["block_size_x"]
params["VECTOR_SIZE"] = 4
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm optimization 1 (double precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

error_D = numpy.abs(answer[0] - D_double).max()

print("    Max error in D:" + str(error_D))

print("Done")
print('---------------------------------------------------------------------\n')

# %% Tiling matrix multiplication algorithm (no bank conflict and computation optimization) (single precision)
# Setup kernel
kernel_name = "matrix_mult_tiling_optimization_1_single"
kernel_source = "../cu/matrix_mult_tiling_optimization_1_single.cu"

problem_size = (n_columns_D, n_rows_D)

arguments = [D_single_kernel, A_single, B_single, n_columns_A, n_columns_B, n_rows_A]

params = dict()
params["block_size_x"] = 16
params["block_size_y"] = 4
params["TILE_SIZE"] = params["block_size_x"]
params["VECTOR_SIZE"] = 4
grid_div_x = ["TILE_SIZE*4"]
grid_div_y = ["TILE_SIZE"]

# %% Run kernel
print('\n---------------------------------------------------------------------')
print('Running tiling algorithm optimization 1 (single precision)...')
answer = run_kernel(kernel_name, kernel_source, problem_size, arguments, params, grid_div_x = grid_div_x, grid_div_y = grid_div_y)

error_D = numpy.abs(answer[0] - D_single).max()

print("    Max error in D:" + str(error_D))

print("Done")
print('---------------------------------------------------------------------\n')



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
