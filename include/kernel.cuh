#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <mma.h>
#include <cstdint>
#include <limits>
#include "param.h"



constexpr uint32_t kInvalidGpuOffset = std::numeric_limits<uint32_t>::max();

struct GpuGraphState {
  uint32_t max_elements{0};
  uint32_t cur_element_count{0};
  uint32_t size_data_per_element{0};
  uint32_t size_links_per_element{0};  // TODO: M + the batch size of new vectors
  uint32_t M{0};
  // uint32_t maxM{0};
  uint32_t maxM0{0};
  uint32_t ef_construction{0};
  uint32_t ef{0};
  uint32_t enterpoint_node{kInvalidGpuOffset};
  uint32_t size_links_level0{0};
  uint32_t offsetData{0};
  uint32_t offsetLevel0{0};
  uint32_t label_offset{0};
  uint32_t data_size{0};
  int32_t maxlevel{-1};
  float mult{0.0f};
  float revSize{0.0f};
  uint32_t vector_dim{0};
  uint32_t vector_data_bytes{0};
  uint32_t label_lookup_size{0};

  // data
  float *vector_data{nullptr};

  // to be updated after construction
  uint32_t link_lists_bytes{0};
  char *level0_links{nullptr};
  half *half_vector_data{nullptr};
  char **link_lists{nullptr};
  float *neighbor_distances{nullptr};
  uint32_t *element_levels{nullptr};
  // runtime states shared by all threads on GPU
  // no need to maintain on CPU
  uint32_t *old_vector_fetch_index{nullptr};
  // Dense old-vector panel for WMMA B: packed_old_vectors[i * vector_dim + d]
  // is component d of old_vector_fetch_index[i]. Sized LEVEL_SZ_THRES * vector_dim.
  half *old_vector_store{nullptr};
  uint32_t old_vec_fetch_offset{0};
  uint32_t *level_counts{nullptr};
  float *vector_powers{nullptr};
  float *precomputed_new_new_dist{nullptr};
  uint32_t *precomputed_new_new_rank{nullptr};
  float *news_dist{nullptr};
  uint32_t *news_rank{nullptr};
  uint32_t *visited{nullptr};
  // Per-block generation counters for `visited`. Persist across batches so a
  // block only memsets its slice when the tag wraps.
  uint32_t *visited_tags{nullptr};
  uint32_t *frozen_link_counts{nullptr};
  uint32_t *changed_old_links{nullptr};
  uint32_t *changed_old_link_counts{nullptr};

#ifdef PROFILE_BUILD_PHASES
  bool profile_build_phases{false};
  uint64_t *build_phase_cycles{nullptr};
  uint64_t build_batch_count{0};
  // search_knn_at_lower: per (batch, block) counters, layout batch * GRID_DIM + blockIdx
  uint64_t *search_block_cycles{nullptr};
  uint64_t *search_block_clear_cycles{nullptr};
  uint64_t *search_block_expands{nullptr};
#endif
};

__global__ void prepare_graph_kernel(GpuGraphState *state);

cudaError_t launch_prepare_graph_kernel(GpuGraphState *state);

cudaError_t launch_precompute_new_new_dist_kernel(GpuGraphState *state, uint32_t max_elements);

cudaError_t launch_build_graph_kernel(GpuGraphState *state);

__device__ void compute_power_kernel(GpuGraphState *state);

__device__ void generate_random_levels_kernel(GpuGraphState *state);

__device__ void copy_float_to_half_kernel(GpuGraphState *state);

__device__ void compute_dist_with_old_kernel(GpuGraphState *state);

__device__ void connect_for_new_kernel(GpuGraphState *state, int startup_level);

// For new vectors
__device__ void prune_candidates_kernel(GpuGraphState *state, int lv);

// For old vectors
__device__ void prune_for_old_kernel(GpuGraphState *state);

__device__ void prune_neighbors_for_all_kernel(GpuGraphState *state);

__device__ void search_knn_at_lower_kernel(GpuGraphState *state, int startup_level);

__device__ void snapshot_frozen_link_counts_kernel(GpuGraphState *state, int max_level);

