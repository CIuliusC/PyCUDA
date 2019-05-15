import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.cublas as cublas

A = np.array(([1, 2, 3], [4, 5, 6]), order = 'F').astype(np.float64)
B = np.array(([7, 8, 1, 5], [9, 10, 0, 9], [11, 12, 5, 5]), order = 'F').astype(np.float64)

A_gpu = gpuarray.to_gpu(A)
B_gpu = gpuarray.to_gpu(B)

m, k = A_gpu.shape
k, n = B_gpu.shape

C_gpu = gpuarray.empty((m, n), np.float64)

alpha=np.float64(1.0)
beta =np.float64(0.0)

cublas_handle = cublas.cublasCreate()
cublas.cublasDgemm(cublas_handle, 'n', 'n', m, n, k, alpha, A_gpu.gpudata, m, B_gpu.gpudata, k, beta, C_gpu.gpudata, m)
cublas.cublasDestroy(cublas_handle)

C_gpu = C_gpu.reshape(C_gpu.shape,order='F')

print(np.dot(A, B))
print(C_gpu)
