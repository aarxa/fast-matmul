// Usage: ./bench [csv_path]
// Times each kernel on square N x N x N problems and reports GFLOP/s
// (2*N^3 floating-point ops per multiply). Takes the best of several runs.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "matmul.hpp"

using Kernel = std::function<void(const float*, const float*, float*, int, int, int)>;

static double time_best(const Kernel& fn, const float* A, const float* B, float* C, int n) {
  // Repeat until we've spent ~0.5 s (at least 3 runs) and keep the fastest.
  double best = 1e30, total = 0.0;
  for (int run = 0; run < 3 || total < 0.5; ++run) {
    const auto t0 = std::chrono::steady_clock::now();
    fn(A, B, C, n, n, n);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best = std::min(best, s);
    total += s;
    if (run >= 50) break;
  }
  return best;
}

int main(int argc, char** argv) {
  struct Entry { const char* name; Kernel fn; int max_n; };
  const std::vector<Entry> kernels = {
      {"naive", mm::naive, 1024},  // too slow beyond this
      {"reorder", mm::reorder, 4096},
      {"tiled", mm::tiled, 4096},
      {"packed", mm::packed, 4096},
      {"threaded", [](const float* A, const float* B, float* C, int M, int N, int K) {
         mm::threaded(A, B, C, M, N, K);
       }, 4096},
      {"accelerate", mm::reference, 4096},
  };
  const int sizes[] = {256, 512, 1024, 2048, 4096};

  FILE* csv = argc > 1 ? std::fopen(argv[1], "w") : nullptr;
  if (csv) std::fprintf(csv, "kernel,n,seconds,gflops\n");

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::printf("%-11s", "N");
  for (auto& k : kernels) std::printf("%12s", k.name);
  std::printf("   (GFLOP/s)\n");

  for (int n : sizes) {
    std::vector<float> A(size_t(n) * n), B(size_t(n) * n), C(size_t(n) * n);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    std::printf("%-11d", n);
    for (auto& k : kernels) {
      if (n > k.max_n) { std::printf("%12s", "-"); continue; }
      const double s = time_best(k.fn, A.data(), B.data(), C.data(), n);
      const double gflops = 2.0 * n * n * n / s / 1e9;
      std::printf("%12.1f", gflops);
      std::fflush(stdout);
      if (csv) std::fprintf(csv, "%s,%d,%.6f,%.2f\n", k.name, n, s, gflops);
    }
    std::printf("\n");
  }
  if (csv) std::fclose(csv);
}
