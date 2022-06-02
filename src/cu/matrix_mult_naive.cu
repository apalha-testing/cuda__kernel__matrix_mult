/**
 * @file matrix_element_matrix_matrix_multiplication_gpu.cu
 *
 * CUDA code to calculate D = A*B
 *
 */


/** Main entry point.
 * Works out where the current thread should read/write to global memory
 * and calls doIterations to do the actual work.
 */
 template<typename REAL>
__global__ void matrix_mult_naive(
                      REAL * D,
                      const REAL * A,
                      const REAL * B,
                      const unsigned int A_B_sum_length,
                      const unsigned int columns_D,
                      const unsigned int rows_D) {

    // Work out which thread we are
    int row_D_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int column_D_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform matrix multiplication
    if (row_D_idx < rows_D && column_D_idx < columns_D){
        REAL tmp_sum = 0.0;
        for (int k_idx = 0; k_idx < A_B_sum_length; k_idx++){
            tmp_sum += A[row_D_idx * A_B_sum_length + k_idx] * B[k_idx * columns_D + column_D_idx];
        }
        // Place in the output array
        D[row_D_idx * columns_D + column_D_idx] = tmp_sum;
    }
}


template __global__ void kernel<float>(float *, const float *, const float *,
                                       cont unsigned int, const unsigned int, const unsigned int);

template __global__ void kernel<double>(double *, const double *, const double *,
                                       cont unsigned int, const unsigned int, const unsigned int);
