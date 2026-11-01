#pragma once
#include <cstddef>

// All kernels compute C = A * B for row-major float matrices.
//   A is M x K, B is K x N, C is M x N.
// C is overwritten (not accumulated into).
namespace mm {

// v1: textbook triple loop (i, j, k). Strided access to B -> cache misses.
void naive(const float* A, const float* B, float* C, int M, int N, int K);

// v2: loop order (i, k, j). Inner loop walks B and C contiguously,
// so the compiler can auto-vectorize it.
void reorder(const float* A, const float* B, float* C, int M, int N, int K);

// v3: v2 + cache blocking so working sets stay in L1/L2.
void tiled(const float* A, const float* B, float* C, int M, int N, int K);

// v4: GotoBLAS-style packing + hand-written 8x8 NEON register micro-kernel.
void packed(const float* A, const float* B, float* C, int M, int N, int K);

// v5: v4 split across threads by rows of C.
void threaded(const float* A, const float* B, float* C, int M, int N, int K,
              int num_threads = 0);  // 0 = std::thread::hardware_concurrency

// Reference: Apple Accelerate cblas_sgemm (vendor-tuned BLAS).
void reference(const float* A, const float* B, float* C, int M, int N, int K);

}  // namespace mm
