#include "matmul.hpp"

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace mm {

void reference(const float* A, const float* B, float* C, int M, int N, int K) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
              1.0f, A, K, B, N, 0.0f, C, N);
}

}  // namespace mm
