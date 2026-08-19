// Microbenchmark: CUDA-core inner product vs reshape-to-matrix + WMMA GEMM + trace.
//
//   CUDA cores:  <a,b> = sum_i a[i] * b[i]
//   MM + trace:  reshape a,b into n x n (n = 16-aligned ceil(sqrt(dim)), zero-padded),
//                C = A * B^T on tensor cores, <a,b> = tr(C)
//
// Dimension is a runtime flag so different shapes can be compared:
//   ./test-ip-vs-mm --dim 128
//   ./test-ip-vs-mm --dim 256 --pairs 8192 --repeats 50

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <fmt/core.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *expr) {
  if (status == cudaSuccess) {
    return;
  }
  std::ostringstream oss;
  oss << expr << " failed: " << cudaGetErrorString(status);
  throw std::runtime_error(oss.str());
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr)

template <typename T>
struct DeviceBuffer {
  T *ptr{nullptr};

  ~DeviceBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
};

struct Options {
  int dim = 256;
  int pairs = 4096;
  int repeats = 20;
  int warmup = 5;
  int matrix_side = 0;  // 0 = infer from dim
  uint32_t seed = 42;
};

void print_usage(const char *argv0) {
  fmt::print(
      "Usage: {} [--dim N] [--pairs N] [--repeats N] [--warmup N] [--side N] [--seed N]\n"
      "  --dim      Vector length (default 256). Padded to a 16-aligned square for MM.\n"
      "  --pairs    Independent random vector pairs (default 4096).\n"
      "  --repeats  Timed kernel launches (default 20).\n"
      "  --warmup   Untimed launches before timing (default 5).\n"
      "  --side     Override MM matrix side; must be a multiple of 16 and side*side >= dim.\n"
      "  --seed     RNG seed (default 42).\n",
      argv0);
}

int parse_positive_int(const char *text, const char *flag) {
  const int value = std::stoi(text);
  if (value <= 0) {
    throw std::runtime_error(std::string(flag) + " must be a positive integer");
  }
  return value;
}

Options parse_args(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
      }
      return argv[++i];
    };

    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--dim") {
      opt.dim = parse_positive_int(need_value("--dim"), "--dim");
    } else if (arg == "--pairs") {
      opt.pairs = parse_positive_int(need_value("--pairs"), "--pairs");
    } else if (arg == "--repeats") {
      opt.repeats = parse_positive_int(need_value("--repeats"), "--repeats");
    } else if (arg == "--warmup") {
      opt.warmup = std::stoi(need_value("--warmup"));
      if (opt.warmup < 0) {
        throw std::runtime_error("--warmup must be >= 0");
      }
    } else if (arg == "--side") {
      opt.matrix_side = parse_positive_int(need_value("--side"), "--side");
    } else if (arg == "--seed") {
      opt.seed = static_cast<uint32_t>(std::stoul(need_value("--seed")));
    } else if (!arg.empty() && arg[0] != '-' && i == 1) {
      opt.dim = parse_positive_int(arg.c_str(), "dim");
    } else {
      throw std::runtime_error("unknown argument: " + arg + " (try --help)");
    }
  }
  return opt;
}

int wmma_aligned_side(int dim) {
  const int side = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(dim))));
  return (side + 15) / 16 * 16;
}

int choose_matrix_side(const Options &opt) {
  if (opt.matrix_side == 0) {
    return wmma_aligned_side(opt.dim);
  }
  if (opt.matrix_side % 16 != 0) {
    throw std::runtime_error("--side must be a multiple of 16 (WMMA tile size)");
  }
  if (static_cast<long long>(opt.matrix_side) * opt.matrix_side < opt.dim) {
    throw std::runtime_error("--side is too small to hold the vector (need side*side >= dim)");
  }
  return opt.matrix_side;
}

std::vector<float> make_random_vectors(int pairs, int dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> vectors(static_cast<size_t>(pairs) * 2 * dim);
  for (float &value : vectors) {
    value = dist(rng);
  }
  return vectors;
}

std::vector<float> cpu_inner_products(const std::vector<float> &vectors, int pairs, int dim) {
  std::vector<float> ips(static_cast<size_t>(pairs));
  for (int pair = 0; pair < pairs; ++pair) {
    const float *a = vectors.data() + static_cast<size_t>(pair) * 2 * dim;
    const float *b = a + dim;
    double acc = 0.0;
    for (int i = 0; i < dim; ++i) {
      acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    ips[static_cast<size_t>(pair)] = static_cast<float>(acc);
  }
  return ips;
}

bool cuda_available(std::string &reason) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    reason = cudaGetErrorString(status);
    cudaGetLastError();
    return false;
  }
  if (device_count <= 0) {
    reason = "no CUDA device detected";
    return false;
  }
  return true;
}

__device__ float block_reduce_sum(float val) {
  constexpr unsigned kMask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(kMask, val, offset);
  }

  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = val;
  }
  __syncthreads();

  const int nwarps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < nwarps) ? warp_sums[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(kMask, val, offset);
    }
  }
  return val;
}

// Point-wise multiply on CUDA cores, then a block reduction of the products.
__global__ void inner_product_cuda_core(const float *vectors, float *out, int dim, int pairs) {
  const int pair = static_cast<int>(blockIdx.x);
  if (pair >= pairs) {
    return;
  }

  const float *a = vectors + static_cast<size_t>(pair) * 2 * dim;
  const float *b = a + dim;

  float acc = 0.0f;
  for (int i = static_cast<int>(threadIdx.x); i < dim; i += static_cast<int>(blockDim.x)) {
    acc += a[i] * b[i];
  }
  acc = block_reduce_sum(acc);
  if (threadIdx.x == 0) {
    out[pair] = acc;
  }
}

// Pack each vector into an n x n row-major half matrix, zero-padding unused entries.
__global__ void pack_vectors_to_matrices(
    const float *vectors, half *a_mat, half *b_mat, int dim, int n, int pairs) {
  const int pair = static_cast<int>(blockIdx.x);
  if (pair >= pairs) {
    return;
  }

  const float *a = vectors + static_cast<size_t>(pair) * 2 * dim;
  const float *b = a + dim;
  half *A = a_mat + static_cast<size_t>(pair) * n * n;
  half *B = b_mat + static_cast<size_t>(pair) * n * n;
  const int matrix_elems = n * n;

  for (int i = static_cast<int>(threadIdx.x); i < matrix_elems; i += static_cast<int>(blockDim.x)) {
    A[i] = (i < dim) ? __float2half(a[i]) : __float2half(0.0f);
    B[i] = (i < dim) ? __float2half(b[i]) : __float2half(0.0f);
  }
}

// C = A * B^T via tensor-core WMMA, then inner product = tr(C).
// Flattening A, B row-major into vectors a, b gives tr(A B^T) = sum_ij A_ij B_ij = a·b.
__global__ void inner_product_mm_trace(const half *a_mat, const half *b_mat, float *out, int n, int pairs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("inner_product_mm_trace requires SM 7.0+ for WMMA\n");
  }
  (void)a_mat;
  (void)b_mat;
  (void)out;
  (void)n;
  (void)pairs;
#else
  namespace wmma = nvcuda::wmma;
  constexpr int kWmma = 16;

  const int pair = static_cast<int>(blockIdx.x);
  if (pair >= pairs) {
    return;
  }

  extern __shared__ __align__(32) float c_smem[];
  const half *A = a_mat + static_cast<size_t>(pair) * n * n;
  const half *B = b_mat + static_cast<size_t>(pair) * n * n;

  const int warp_id = static_cast<int>(threadIdx.x) / warpSize;
  const int warp_count = static_cast<int>(blockDim.x) / warpSize;
  const int tiles = n / kWmma;
  const int tile_count = tiles * tiles;

  if (warp_id < warp_count) {
    for (int tile = warp_id; tile < tile_count; tile += warp_count) {
      const int tile_row = tile / tiles;
      const int tile_col = tile % tiles;
      const int row0 = tile_row * kWmma;
      const int col0 = tile_col * kWmma;

      wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);

      for (int k = 0; k < n; k += kWmma) {
        wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, half, wmma::col_major> b_frag;
        const half *a_ptr = A + row0 * n + k;
        const half *b_ptr = B + col0 * n + k;
        wmma::load_matrix_sync(a_frag, a_ptr, n);
        wmma::load_matrix_sync(b_frag, b_ptr, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      wmma::store_matrix_sync(c_smem + row0 * n + col0, c_frag, n, wmma::mem_row_major);
    }
  }
  __syncthreads();

  float acc = 0.0f;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    acc += c_smem[i * n + i];
  }
  acc = block_reduce_sum(acc);
  if (threadIdx.x == 0) {
    out[pair] = acc;
  }
#endif
}

int choose_block_dim(int work_items) {
  int threads = 32;
  while (threads < 256 && threads < work_items) {
    threads *= 2;
  }
  return threads;
}

float max_abs_error(const std::vector<float> &ref, const std::vector<float> &got) {
  float max_err = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    max_err = std::max(max_err, std::fabs(ref[i] - got[i]));
  }
  return max_err;
}

float max_rel_error(const std::vector<float> &ref, const std::vector<float> &got) {
  float max_err = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float scale = std::max(1.0f, std::fabs(ref[i]));
    max_err = std::max(max_err, std::fabs(ref[i] - got[i]) / scale);
  }
  return max_err;
}

template <typename Launch>
float time_launches(Launch &&launch, int warmup, int repeats) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms;
}

void require_close(
    const std::vector<float> &ref,
    const std::vector<float> &got,
    float rel_tol,
    float abs_tol,
    const char *label) {
  if (got.size() != ref.size()) {
    throw std::runtime_error(std::string(label) + " size mismatch");
  }
  for (size_t i = 0; i < ref.size(); ++i) {
    const float err = std::fabs(ref[i] - got[i]);
    const float scale = std::max(1.0f, std::fabs(ref[i]));
    if (err <= abs_tol || err <= rel_tol * scale) {
      continue;
    }
    std::ostringstream oss;
    oss << label << " mismatch at pair " << i << ": expected " << ref[i] << ", got " << got[i]
        << " (abs err " << err << ")";
    throw std::runtime_error(oss.str());
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    std::string skip_reason;
    if (!cuda_available(skip_reason)) {
      fmt::print("[SKIP] {}\n", skip_reason);
      return 0;
    }

    const Options opt = parse_args(argc, argv);
    const int n = choose_matrix_side(opt);
    const int dim = opt.dim;
    const int pairs = opt.pairs;
    const size_t vector_count = static_cast<size_t>(pairs) * 2 * dim;
    const size_t matrix_elems = static_cast<size_t>(pairs) * n * n;
    const size_t smem_bytes = static_cast<size_t>(n) * n * sizeof(float);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (smem_bytes > static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
      throw std::runtime_error(fmt::format(
          "MM result matrix {}x{} needs {} bytes of shared memory, device max is {}",
          n,
          n,
          smem_bytes,
          prop.sharedMemPerBlockOptin));
    }

    fmt::print(
        "dim={}  pairs={}  mm_matrix={}x{}  ({} padded zeros / vector)  device={}\n",
        dim,
        pairs,
        n,
        n,
        n * n - dim,
        prop.name);

    const std::vector<float> host_vectors = make_random_vectors(pairs, dim, opt.seed);
    const std::vector<float> cpu_ips = cpu_inner_products(host_vectors, pairs, dim);

    DeviceBuffer<float> d_vectors;
    DeviceBuffer<float> d_ip_core;
    DeviceBuffer<float> d_ip_mm;
    DeviceBuffer<half> d_a_mat;
    DeviceBuffer<half> d_b_mat;

    CUDA_CHECK(cudaMalloc(&d_vectors.ptr, vector_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ip_core.ptr, static_cast<size_t>(pairs) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ip_mm.ptr, static_cast<size_t>(pairs) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a_mat.ptr, matrix_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b_mat.ptr, matrix_elems * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(
        d_vectors.ptr, host_vectors.data(), vector_count * sizeof(float), cudaMemcpyHostToDevice));

    const int core_block = choose_block_dim(dim);
    const int pack_block = choose_block_dim(n * n);
    const int mm_tiles = (n / 16) * (n / 16);
    const int mm_block = choose_block_dim(mm_tiles * 32);

    if (smem_bytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
      CUDA_CHECK(cudaFuncSetAttribute(
          inner_product_mm_trace, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }

    pack_vectors_to_matrices<<<pairs, pack_block>>>(d_vectors.ptr, d_a_mat.ptr, d_b_mat.ptr, dim, n, pairs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    inner_product_cuda_core<<<pairs, core_block>>>(d_vectors.ptr, d_ip_core.ptr, dim, pairs);
    CUDA_CHECK(cudaGetLastError());
    inner_product_mm_trace<<<pairs, mm_block, smem_bytes>>>(d_a_mat.ptr, d_b_mat.ptr, d_ip_mm.ptr, n, pairs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_core(static_cast<size_t>(pairs));
    std::vector<float> gpu_mm(static_cast<size_t>(pairs));
    CUDA_CHECK(cudaMemcpy(
        gpu_core.data(), d_ip_core.ptr, static_cast<size_t>(pairs) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        gpu_mm.data(), d_ip_mm.ptr, static_cast<size_t>(pairs) * sizeof(float), cudaMemcpyDeviceToHost));

    require_close(cpu_ips, gpu_core, 1.0e-5f, 1.0e-5f, "CUDA-core inner product");
    require_close(cpu_ips, gpu_mm, 5.0e-2f, 5.0e-2f, "MM+trace inner product");

    const float core_ms = time_launches(
        [&]() { inner_product_cuda_core<<<pairs, core_block>>>(d_vectors.ptr, d_ip_core.ptr, dim, pairs); },
        opt.warmup,
        opt.repeats);
    const float mm_ms = time_launches(
        [&]() {
          inner_product_mm_trace<<<pairs, mm_block, smem_bytes>>>(
              d_a_mat.ptr, d_b_mat.ptr, d_ip_mm.ptr, n, pairs);
        },
        opt.warmup,
        opt.repeats);

    const double core_us_per_pair = (core_ms * 1000.0) / (static_cast<double>(opt.repeats) * pairs);
    const double mm_us_per_pair = (mm_ms * 1000.0) / (static_cast<double>(opt.repeats) * pairs);

    fmt::print("[PASS] CUDA-core IP  max abs err {:.3e}  max rel err {:.3e}\n",
               max_abs_error(cpu_ips, gpu_core),
               max_rel_error(cpu_ips, gpu_core));
    fmt::print("[PASS] MM+trace IP   max abs err {:.3e}  max rel err {:.3e}  (fp16 WMMA)\n",
               max_abs_error(cpu_ips, gpu_mm),
               max_rel_error(cpu_ips, gpu_mm));
    fmt::print(
        "CUDA-core IP:  {:8.3f} ms  ({:.4f} us/pair, {} repeats)\n", core_ms, core_us_per_pair, opt.repeats);
    fmt::print(
        "MM+trace WMMA: {:8.3f} ms  ({:.4f} us/pair, {} repeats)\n", mm_ms, mm_us_per_pair, opt.repeats);
    fmt::print("speedup (core / mm): {:.3f}x  (>1 means MM+trace is faster)\n", core_ms / mm_ms);
    return 0;
  } catch (const std::exception &ex) {
    fmt::print(stderr, "[FAIL] {}\n", ex.what());
    return 1;
  }
}
