// v4/v5: GotoBLAS-style GEMM.
//
// Loop nest (outer to inner):
//   jc over N in NC-wide slabs   -> B slab
//   pc over K in KC-deep slabs   -> pack B[pc:pc+KC, jc:jc+NC] into NR-wide panels (lives in L2)
//   ic over M in MC-tall slabs   -> pack A[ic:ic+MC, pc:pc+KC] into MR-tall panels (lives in L2)
//   jr, ir over the packed panels -> 8x8 micro-kernel, accumulators in registers
//
// Packing turns strided matrix reads into one contiguous stream per panel,
// and zero-pads ragged edges so the micro-kernel never needs bounds checks.
#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "matmul.hpp"

namespace mm {
namespace {

constexpr int MR = 8, NR = 8;                   // micro-tile held in registers
constexpr int MC = 128, KC = 256, NC = 1024;    // cache blocking

// Ap layout: [panel][k][MR]
void pack_A(const float* A, int K, int i0, int mc, int p0, int kc, float* Ap) {
  for (int ir = 0; ir < mc; ir += MR)
    for (int k = 0; k < kc; ++k)
      for (int r = 0; r < MR; ++r)
        *Ap++ = (ir + r < mc) ? A[(i0 + ir + r) * K + p0 + k] : 0.0f;
}

// Bp layout: [panel][k][NR]
void pack_B(const float* B, int N, int p0, int kc, int j0, int nc, float* Bp) {
  for (int jr = 0; jr < nc; jr += NR)
    for (int k = 0; k < kc; ++k) {
      const float* row = B + (p0 + k) * N + j0 + jr;
      if (jr + NR <= nc) {
        std::memcpy(Bp, row, NR * sizeof(float));
      } else {
        for (int c = 0; c < NR; ++c) Bp[c] = (jr + c < nc) ? row[c] : 0.0f;
      }
      Bp += NR;
    }
}

// C[0:mr, 0:nr] += Ap_panel * Bp_panel over kc steps.
void micro_kernel(int kc, const float* a, const float* b, float* C, int ldc, int mr, int nr) {
#if defined(__ARM_NEON)
  // 8 rows x 2 vectors of 4 = 16 of the 32 NEON registers hold the C tile.
  float32x4_t c[MR][2];
  for (int r = 0; r < MR; ++r) c[r][0] = c[r][1] = vdupq_n_f32(0.0f);

  for (int k = 0; k < kc; ++k) {
    const float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4);
    const float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
    // Outer product a(8) x b(8): one fused multiply-add per (row, half-row).
#define FMA_ROW(r, av, lane)                              \
  c[r][0] = vfmaq_laneq_f32(c[r][0], b0, av, lane);       \
  c[r][1] = vfmaq_laneq_f32(c[r][1], b1, av, lane);
    FMA_ROW(0, a0, 0) FMA_ROW(1, a0, 1) FMA_ROW(2, a0, 2) FMA_ROW(3, a0, 3)
    FMA_ROW(4, a1, 0) FMA_ROW(5, a1, 1) FMA_ROW(6, a1, 2) FMA_ROW(7, a1, 3)
#undef FMA_ROW
    a += MR;
    b += NR;
  }

  if (mr == MR && nr == NR) {
    for (int r = 0; r < MR; ++r) {
      float* row = C + r * ldc;
      vst1q_f32(row, vaddq_f32(vld1q_f32(row), c[r][0]));
      vst1q_f32(row + 4, vaddq_f32(vld1q_f32(row + 4), c[r][1]));
    }
    return;
  }
  float tmp[MR][NR];
  for (int r = 0; r < MR; ++r) {
    vst1q_f32(tmp[r], c[r][0]);
    vst1q_f32(tmp[r] + 4, c[r][1]);
  }
#else
  float tmp[MR][NR] = {};
  for (int k = 0; k < kc; ++k, a += MR, b += NR)
    for (int r = 0; r < MR; ++r)
      for (int j = 0; j < NR; ++j) tmp[r][j] += a[r] * b[j];
#endif
  for (int r = 0; r < mr; ++r)
    for (int j = 0; j < nr; ++j) C[r * ldc + j] += tmp[r][j];
}

// Computes rows [m0, m1) of C.
void gemm_rows(const float* A, const float* B, float* C, int N, int K, int m0, int m1) {
  std::memset(C + m0 * N, 0, sizeof(float) * (m1 - m0) * N);
  std::vector<float> Ap(MC * KC), Bp(KC * ((NC + NR - 1) / NR * NR));

  for (int jc = 0; jc < N; jc += NC) {
    const int nc = std::min(NC, N - jc);
    for (int pc = 0; pc < K; pc += KC) {
      const int kc = std::min(KC, K - pc);
      pack_B(B, N, pc, kc, jc, nc, Bp.data());
      for (int ic = m0; ic < m1; ic += MC) {
        const int mc = std::min(MC, m1 - ic);
        pack_A(A, K, ic, mc, pc, kc, Ap.data());
        for (int jr = 0; jr < nc; jr += NR)
          for (int ir = 0; ir < mc; ir += MR)
            micro_kernel(kc, Ap.data() + ir * kc, Bp.data() + jr * kc,
                         C + (ic + ir) * N + jc + jr, N,
                         std::min(MR, mc - ir), std::min(NR, nc - jr));
      }
    }
  }
}

}  // namespace

void packed(const float* A, const float* B, float* C, int M, int N, int K) {
  gemm_rows(A, B, C, N, K, 0, M);
}

void threaded(const float* A, const float* B, float* C, int M, int N, int K, int num_threads) {
  if (num_threads <= 0) num_threads = std::max(1u, std::thread::hardware_concurrency());
  // Hand each thread a contiguous, MR-aligned band of rows.
  const int blocks = (M + MR - 1) / MR;
  num_threads = std::min(num_threads, blocks);
  std::vector<std::thread> pool;
  for (int t = 0; t < num_threads; ++t) {
    const int m0 = std::min(M, blocks * t / num_threads * MR);
    const int m1 = std::min(M, blocks * (t + 1) / num_threads * MR);
    if (m0 < m1) pool.emplace_back(gemm_rows, A, B, C, N, K, m0, m1);
  }
  for (auto& th : pool) th.join();
}

}  // namespace mm
