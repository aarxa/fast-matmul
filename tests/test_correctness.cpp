// Checks every kernel against the vendor BLAS on awkward shapes
// (non-square, not multiples of the tile sizes, tiny, and larger than one block).
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "matmul.hpp"

using Kernel = std::function<void(const float*, const float*, float*, int, int, int)>;

int main() {
  const std::vector<std::pair<const char*, Kernel>> kernels = {
      {"naive", mm::naive},
      {"reorder", mm::reorder},
      {"tiled", mm::tiled},
      {"packed", mm::packed},
      {"threaded", [](const float* A, const float* B, float* C, int M, int N, int K) {
         mm::threaded(A, B, C, M, N, K);
       }},
  };
  const int shapes[][3] = {{1, 1, 1},     {7, 5, 3},       {8, 8, 8},     {33, 17, 65},
                           {100, 200, 50}, {129, 257, 300}, {512, 512, 512}, {300, 1100, 270}};

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  int failures = 0;

  for (auto& s : shapes) {
    const int M = s[0], N = s[1], K = s[2];
    std::vector<float> A(M * K), B(K * N), ref(M * N), C(M * N);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    mm::reference(A.data(), B.data(), ref.data(), M, N, K);

    for (auto& [name, fn] : kernels) {
      std::fill(C.begin(), C.end(), NAN);  // catch any element a kernel forgets to write
      fn(A.data(), B.data(), C.data(), M, N, K);
      float max_err = 0.0f;
      for (int i = 0; i < M * N; ++i) {
        const float err = std::fabs(C[i] - ref[i]);
        max_err = std::isnan(C[i]) ? INFINITY : std::max(max_err, err);
      }
      // Float sums of K terms in different orders differ by ~K * eps.
      const float tol = 1e-5f * K + 1e-5f;
      const bool ok = max_err <= tol;
      failures += !ok;
      std::printf("%-9s %4dx%4dx%4d  max_err=%.2e  %s\n", name, M, N, K, max_err, ok ? "ok" : "FAIL");
    }
  }
  std::printf(failures ? "\n%d FAILURES\n" : "\nall passed\n", failures);
  return failures ? 1 : 0;
}
