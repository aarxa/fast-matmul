#include <algorithm>
#include <cstring>

#include "matmul.hpp"

namespace mm {

void naive(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];  // B column walk: stride N
      C[i * N + j] = acc;
    }
}

void reorder(const float* A, const float* B, float* C, int M, int N, int K) {
  std::memset(C, 0, sizeof(float) * M * N);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) {
      const float a = A[i * K + k];
      const float* b = B + k * N;
      float* c = C + i * N;
      for (int j = 0; j < N; ++j) c[j] += a * b[j];  // unit stride -> vectorizes
    }
}

// Block sizes chosen so a TK x TN tile of B (64 KB) fits in L2 comfortably
// and one TN-wide row of C stays hot in L1 while we sweep k.
constexpr int TM = 64, TN = 256, TK = 64;

void tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  std::memset(C, 0, sizeof(float) * M * N);
  for (int i0 = 0; i0 < M; i0 += TM)
    for (int k0 = 0; k0 < K; k0 += TK)
      for (int j0 = 0; j0 < N; j0 += TN) {
        const int i1 = std::min(i0 + TM, M), k1 = std::min(k0 + TK, K), j1 = std::min(j0 + TN, N);
        for (int i = i0; i < i1; ++i)
          for (int k = k0; k < k1; ++k) {
            const float a = A[i * K + k];
            const float* b = B + k * N;
            float* c = C + i * N;
            for (int j = j0; j < j1; ++j) c[j] += a * b[j];
          }
      }
}

}  // namespace mm
