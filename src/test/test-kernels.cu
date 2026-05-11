#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "kernel.cuh"

namespace {

constexpr uint32_t kVectorDim = DIM;
constexpr uint32_t kTrackedLevels = 128;

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

struct PreparedGraphResult {
  std::vector<float> vector_powers;
  std::vector<int32_t> element_levels;
  std::vector<uint32_t> level_counts;
  int32_t maxlevel{-1};
};

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<float> make_test_vectors(uint32_t max_elements) {
  std::vector<float> vectors(static_cast<size_t>(max_elements) * kVectorDim);
  for (uint32_t row = 0; row < max_elements; ++row) {
    for (uint32_t col = 0; col < kVectorDim; ++col) {
      const float centered = static_cast<float>((static_cast<int>(row * 17 + col * 13) % 29) - 14);
      vectors[static_cast<size_t>(row) * kVectorDim + col] = centered * 0.125f + static_cast<float>(row % 7) * 0.01f;
    }
  }
  return vectors;
}

std::vector<float> compute_cpu_powers(const std::vector<float> &vectors, uint32_t max_elements) {
  std::vector<float> powers(max_elements, 0.0f);
  for (uint32_t row = 0; row < max_elements; ++row) {
    float power = 0.0f;
    const size_t row_offset = static_cast<size_t>(row) * kVectorDim;
    for (uint32_t col = 0; col < kVectorDim; ++col) {
      const float value = vectors[row_offset + col];
      power += value * value;
    }
    powers[row] = power;
  }
  return powers;
}

PreparedGraphResult run_prepare_graph(const std::vector<float> &vectors, uint32_t max_elements, float rev_size) {
  require(vectors.size() == static_cast<size_t>(max_elements) * kVectorDim, "unexpected vector buffer size");

  DeviceBuffer<float> d_vectors;
  DeviceBuffer<float> d_vector_powers;
  DeviceBuffer<uint32_t> d_element_levels;
  DeviceBuffer<uint32_t> d_level_counts;
  DeviceBuffer<half> d_half_vectors;
  DeviceBuffer<GpuGraphState> d_state;

  const size_t vector_count = static_cast<size_t>(max_elements) * kVectorDim;
  const size_t vector_bytes = vector_count * sizeof(float);

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vectors.ptr), vector_bytes));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_vector_powers.ptr), static_cast<size_t>(max_elements) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void **>(&d_element_levels.ptr), static_cast<size_t>(max_elements) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void **>(&d_level_counts.ptr), static_cast<size_t>(kTrackedLevels) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_half_vectors.ptr), vector_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_state.ptr), sizeof(GpuGraphState)));

  CUDA_CHECK(cudaMemcpy(d_vectors.ptr, vectors.data(), vector_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_vector_powers.ptr, 0, static_cast<size_t>(max_elements) * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_element_levels.ptr, 0, static_cast<size_t>(max_elements) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_level_counts.ptr, 0, static_cast<size_t>(kTrackedLevels) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_half_vectors.ptr, 0, vector_count * sizeof(half)));

  GpuGraphState host_state{};
  host_state.max_elements = max_elements;
  host_state.vector_dim = kVectorDim;
  host_state.revSize = rev_size;
  host_state.maxlevel = -1;
  host_state.vector_data = (d_vectors.ptr);
  host_state.vector_powers = d_vector_powers.ptr;
  host_state.element_levels = d_element_levels.ptr;
  host_state.level_counts = d_level_counts.ptr;
  host_state.half_vector_data = d_half_vectors.ptr;

  CUDA_CHECK(cudaMemcpy(d_state.ptr, &host_state, sizeof(GpuGraphState), cudaMemcpyHostToDevice));
  CUDA_CHECK(launch_prepare_graph_kernel(d_state.ptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  PreparedGraphResult result;
  result.vector_powers.resize(max_elements);
  result.element_levels.resize(max_elements);
  result.level_counts.resize(kTrackedLevels);

  CUDA_CHECK(cudaMemcpy(
      result.vector_powers.data(),
      d_vector_powers.ptr,
      static_cast<size_t>(max_elements) * sizeof(float),
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      result.element_levels.data(),
      d_element_levels.ptr,
      static_cast<size_t>(max_elements) * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      result.level_counts.data(),
      d_level_counts.ptr,
      static_cast<size_t>(kTrackedLevels) * sizeof(uint32_t),
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&host_state, d_state.ptr, sizeof(GpuGraphState), cudaMemcpyDeviceToHost));
  result.maxlevel = host_state.maxlevel;

  return result;
}

bool almost_equal(float lhs, float rhs) {
  const float scale = std::max(1.0f, std::max(std::fabs(lhs), std::fabs(rhs)));
  return std::fabs(lhs - rhs) <= 1.0e-5f * scale;
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

void test_compute_power_kernel() {
  constexpr uint32_t kMaxElements = 9;

  const std::vector<float> vectors = make_test_vectors(kMaxElements);
  const std::vector<float> expected = compute_cpu_powers(vectors, kMaxElements);
  const PreparedGraphResult result = run_prepare_graph(vectors, kMaxElements, 1.5f);

  require(result.vector_powers.size() == expected.size(), "power result size mismatch");
  for (uint32_t i = 0; i < kMaxElements; ++i) {
    if (almost_equal(result.vector_powers[i], expected[i])) {
      continue;
    }

    std::ostringstream oss;
    oss << "compute_power_kernel mismatch at vector " << i << ": expected " << expected[i] << ", got "
        << result.vector_powers[i];
    throw std::runtime_error(oss.str());
  }
}

void test_generate_random_levels_kernel() {
  constexpr uint32_t kMaxElements = 513;

  const std::vector<float> vectors = make_test_vectors(kMaxElements);
  const PreparedGraphResult result = run_prepare_graph(vectors, kMaxElements, 1.5f);

  require(result.element_levels.size() == kMaxElements, "element level size mismatch");
  require(result.level_counts.size() == kTrackedLevels, "level count size mismatch");

  std::vector<uint32_t> expected_level_counts(kTrackedLevels, 0);
  int32_t expected_maxlevel = 0;

  for (uint32_t i = 0; i < kMaxElements; ++i) {
    const int32_t level = result.element_levels[i];
    require(level >= 0, "generate_random_levels_kernel produced a negative level");
    require(level < static_cast<int32_t>(kTrackedLevels), "generated level exceeds tracked test range");

    expected_maxlevel = std::max(expected_maxlevel, level);
    for (int32_t lower_level = 0; lower_level <= level; ++lower_level) {
      expected_level_counts[static_cast<size_t>(lower_level)]++;
    }
  }

  require(result.level_counts[0] == kMaxElements, "level_counts[0] should include every element");
  require(result.maxlevel == expected_maxlevel, "maxlevel does not match the generated levels");

  for (uint32_t level = 0; level < kTrackedLevels; ++level) {
    if (result.level_counts[level] != expected_level_counts[level]) {
      std::ostringstream oss;
      oss << "level_counts mismatch at level " << level << ": expected " << expected_level_counts[level] << ", got "
          << result.level_counts[level];
      throw std::runtime_error(oss.str());
    }
  }

  for (uint32_t level = 1; level < kTrackedLevels; ++level) {
    if (result.level_counts[level - 1] >= result.level_counts[level]) {
      continue;
    }

    std::ostringstream oss;
    oss << "level_counts must be non-increasing, but level " << (level - 1) << " has " << result.level_counts[level - 1]
        << " and level " << level << " has " << result.level_counts[level];
    throw std::runtime_error(oss.str());
  }
}

}  // namespace

int main() {
  try {
    std::string skip_reason;
    if (!cuda_available(skip_reason)) {
      std::cout << "[SKIP] " << skip_reason << std::endl;
      return 0;
    }

    test_compute_power_kernel();
    std::cout << "[PASS] compute_power_kernel" << std::endl;

    test_generate_random_levels_kernel();
    std::cout << "[PASS] generate_random_levels_kernel" << std::endl;
  } catch (const std::exception &ex) {
    std::cerr << "[FAIL] " << ex.what() << std::endl;
    return 1;
  }

  return 0;
}
