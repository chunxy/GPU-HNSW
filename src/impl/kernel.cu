#include <stdio.h>
#include <type_traits>
#include "hnswalg-lite.h"
#include "kernel.cuh"

namespace {

__device__ uint64_t build_phase_begin(GpuGraphState *state) {
#ifdef PROFILE_BUILD_PHASES
  return (state->profile_build_phases && blockIdx.x == 0 && threadIdx.x == 0) ? clock64() : 0ULL;
#else
  (void)state;
  return 0ULL;
#endif
}

__device__ void build_phase_record(GpuGraphState *state, BuildPhase phase, uint64_t t0) {
#ifdef PROFILE_BUILD_PHASES
  if (state->profile_build_phases && blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(reinterpret_cast<unsigned long long *>(state->build_phase_cycles + phase), clock64() - t0);
  }
#else
  (void)state;
  (void)phase;
  (void)t0;
#endif
}

__device__ int compute_startup_level(GpuGraphState *state) {
  int lo = -1, hi = state->maxlevel + 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (state->level_counts[mid] <= LEVEL_SZ_THRES) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return min(hi, state->maxlevel);
}

constexpr unsigned long long kRandomLevelSeed = 0x9E3779B97F4A7C15ULL;
constexpr uint32_t kChangedOldLinkFlag = 1U << 31;
constexpr uint32_t kLinkCountMask = ~kChangedOldLinkFlag;
}  // namespace

struct Neighbor {
  float distance;
  int nodeid;
  bool checked;
};

template <typename T>
__device__ inline void swap(T &a, T &b) {
  static_assert(
      std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>,
      "swap(T&, T&) requires T to be copy constructible and copy assignable");
  T t = a;
  a = b;
  b = t;
}

__device__ inline unsigned next_power_of_two(unsigned n) {
  if (n <= 1) return n;
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

// Luckily they are uint32_t.
using hnswlib::linklistsizeint;
using hnswlib::tableint;

__device__ uint32_t *get_linklist(GpuGraphState *state, uint32_t internal_id, int level) {
  return (uint32_t *)(state->link_lists[internal_id] + (level - 1) * state->size_links_per_element);
}

__device__ float *get_linklist_dist(GpuGraphState *state, uint32_t internal_id, int level) {
  return (float *)(state->link_lists[internal_id] + (level - 1) * state->size_links_per_element +
                   sizeof(linklistsizeint) + (state->M + BATCHSZ_PER_NEW) * sizeof(uint32_t));
}

__device__ char *level0_link_bytes(GpuGraphState *state, uint32_t internal_id) {
  // size_links_level0 is ~4 KiB; uint32_t id * size wraps past ~1e6 vectors.
  return state->level0_links + static_cast<size_t>(internal_id) * static_cast<size_t>(state->size_links_level0);
}

__device__ uint32_t *get_linklist0(GpuGraphState *state, uint32_t internal_id) {
  return (uint32_t *)level0_link_bytes(state, internal_id);
}

__device__ float *get_linklist_dist0(GpuGraphState *state, uint32_t internal_id) {
  return (float *)(level0_link_bytes(state, internal_id) + sizeof(linklistsizeint) +
                   (state->maxM0 + BATCHSZ_PER_NEW) * sizeof(uint32_t));
}

__device__ uint32_t *get_level_linklist(GpuGraphState *state, uint32_t internal_id, int level) {
  return level == 0 ? get_linklist0(state, internal_id) : get_linklist(state, internal_id, level);
}

__device__ float *get_level_linklist_dist(GpuGraphState *state, uint32_t internal_id, int level) {
  return level == 0 ? get_linklist_dist0(state, internal_id) : get_linklist_dist(state, internal_id, level);
}

__device__ uint32_t getListCount(uint32_t *ptr) { return *((uint32_t *)ptr); }

__device__ void setListCount(uint32_t *ptr, uint32_t size) { *((tableint *)ptr) = size; }

__device__ uint32_t frozen_link_count_offset(GpuGraphState *state, uint32_t internal_id, int level) {
  return level * state->max_elements + internal_id;
}

__device__ uint32_t get_frozen_link_count(GpuGraphState *state, uint32_t internal_id, int level) {
  if (level < 0 || level >= MAX_HNSW_LEVEL || internal_id >= state->max_elements ||
      state->element_levels[internal_id] < level) {
    return 0;
  }
  return state->frozen_link_counts[frozen_link_count_offset(state, internal_id, level)] & kLinkCountMask;
}

__device__ uint32_t changed_old_link_level_offset(GpuGraphState *state, int level) {
  return level * BATCHSZ_PER_NEW * state->maxM0;
}

__device__ void record_changed_old_link(GpuGraphState *state, uint32_t internal_id, int level) {
  if (internal_id >= state->cur_element_count || level < 0 || level >= MAX_HNSW_LEVEL) return;

  uint32_t *frozen_count = state->frozen_link_counts + frozen_link_count_offset(state, internal_id, level);
  const uint32_t previous = atomicOr(frozen_count, kChangedOldLinkFlag);
  if ((previous & kChangedOldLinkFlag) != 0) return;

  const uint32_t offset = atomicAdd(&state->changed_old_link_counts[level], 1U);
#ifndef NDEBUG
  if (offset >= BATCHSZ_PER_NEW * state->maxM0) {
    printf("Fatal: changed_old_links out of bound at level %d\n", level);
    assert(false);
  }
#endif
  state->changed_old_links[changed_old_link_level_offset(state, level) + offset] = internal_id;
}

#define mul(x, y) (x * y)
#define add(x, y) (x + y)
#define sub(x, y) (x - y)
#define gt(x, y) (x > y)
#define ge(x, y) (x >= y)
#define lt(x, y) (x < y)
#define le(x, y) (x <= y)

__device__ void MaxPqPop(Neighbor *pq, int *size) {
  if (*size == 0) return;
  (*size)--;
  float tail_dist = pq[*size].distance;
  int p = 0, r = 1;
  while (r < *size) {
    if (r < (*size) - 1 && gt(pq[r + 1].distance, pq[r].distance)) r++;
    if (ge(tail_dist, pq[r].distance)) break;
    pq[p] = pq[r];
    p = r;
    r = 2 * p + 1;
  }
  pq[p] = pq[*size];
}

__device__ void MinPqPop(Neighbor *pq, int *size, Neighbor *tmp) {
  if (*size == 0) return;
  (*size)--;
  tmp->distance = pq[0].distance;
  tmp->nodeid = pq[0].nodeid;
  tmp->checked = pq[0].checked;
  float tail_dist = pq[*size].distance;
  int p = 0, r = 1;
  while (r < *size) {
    if (r < (*size) - 1 && lt(pq[r + 1].distance, pq[r].distance)) r++;
    if (le(tail_dist, pq[r].distance)) break;
    pq[p] = pq[r];
    p = r;
    r = 2 * p + 1;
  }
  pq[p] = pq[*size];
}

__device__ void MaxPqPush(Neighbor *pq, int *size, float dist, int nodeid, bool check) {
  int idx = *size;
  while (idx > 0) {
    int nidx = (idx + 1) / 2 - 1;
    if (ge(pq[nidx].distance, dist)) break;
    pq[idx] = pq[nidx];
    idx = nidx;
  }
  pq[idx].distance = dist;
  pq[idx].nodeid = nodeid;
  pq[idx].checked = check;
  (*size)++;
}

__device__ void MinPqPush(Neighbor *pq, int *size, float dist, int nodeid, bool check) {
  int idx = *size;
  while (idx > 0) {
    int nidx = (idx + 1) / 2 - 1;
    if (le(pq[nidx].distance, dist)) break;
    pq[idx] = pq[nidx];
    idx = nidx;
  }
  pq[idx].distance = dist;
  pq[idx].nodeid = nodeid;
  pq[idx].checked = check;
  (*size)++;
}

__device__ void find_closest_in_topq(const Neighbor *topq, int topq_sz, uint32_t *out_id, float *out_dist) {
  if (topq_sz <= 0) {
    return;
  }
  int best = 0;
  for (int i = 1; i < topq_sz; ++i) {
    if (topq[i].distance < topq[best].distance) {
      best = i;
    }
  }
  *out_id = topq[best].nodeid;
  *out_dist = topq[best].distance;
}

__device__ void merge_staged_neighbors_into_queues(
    Neighbor *candq,
    int *candq_sz,
    Neighbor *topq,
    int *topq_sz,
    float *topq_max,
    Neighbor (*warp_staging)[WARP_STAGING_CAP],
    const int *warp_staging_sz,
    const int ef_construction) {
  for (int w = 0; w < SEARCH_WARP_COUNT; ++w) {
    for (int i = 0; i < warp_staging_sz[w]; ++i) {
      const float dist = warp_staging[w][i].distance;
      const int nodeid = warp_staging[w][i].nodeid;
      if (*topq_sz < ef_construction || dist < *topq_max) {
        if (*candq_sz < CANDQ_SZ) {
          MinPqPush(candq, candq_sz, dist, nodeid, false);
        }
        if (*topq_sz < TOPQ_SZ) {
          MaxPqPush(topq, topq_sz, dist, nodeid, true);
        }
        while (*topq_sz > ef_construction) {
          MaxPqPop(topq, topq_sz);
        }
        if (*topq_sz >= ef_construction) {
          *topq_max = topq[0].distance;
        }
      }
    }
  }
}

__device__ void bitonic_sort_id_by_dis(GpuGraphState *state) {
  int len = state->old_vec_fetch_offset;
  float *distances = state->news_dist;
  unsigned *ids = state->news_rank;
  if (len <= 1) return;
  const unsigned sort_len = next_power_of_two(len);
  const int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  for (int vid = blockIdx.x; vid < new_count; vid += gridDim.x) {
    float *target = distances + vid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
    unsigned *target_ids = ids + vid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
    const unsigned tid = threadIdx.x;
    for (unsigned stride = 1; stride < sort_len; stride <<= 1) {
      for (unsigned step = stride; step > 0; step >>= 1) {
        for (unsigned k = tid; k < sort_len / 2; k += blockDim.x) {
          unsigned a = 2 * step * (k / step);
          unsigned b = k % step;
          unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
          unsigned d = a + b + step;
          if (d < len && target[u] > target[d]) {
            swap(target[u], target[d]);
            swap(target_ids[u], target_ids[d]);
          }
        }
        __syncthreads();
      }
    }
  }
}

__device__ void bitonic_sort(GpuGraphState *state, uint32_t vid, int lv) {
  uint32_t *linkl = get_level_linklist(state, vid, lv);
  uint32_t *datal = (uint32_t *)(linkl + 1);
  float *distl = get_level_linklist_dist(state, vid, lv);
  int len = getListCount(linkl);
  if (len <= 1) return;

  const unsigned tid = threadIdx.x;
  const unsigned sort_len = next_power_of_two(len);
  for (unsigned stride = 1; stride < sort_len; stride <<= 1) {
    for (unsigned step = stride; step > 0; step >>= 1) {
      for (unsigned k = tid; k < sort_len / 2; k += blockDim.x) {
        unsigned a = 2 * step * (k / step);
        unsigned b = k % step;
        unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
        unsigned d = a + b + step;
        if (d < len && distl[u] > distl[d]) {
          swap(distl[u], distl[d]);
          swap(datal[u], datal[d]);
        }
      }
      __syncthreads();
    }
  }
}

__device__ void bitonic_sort_id_for_ll(GpuGraphState *state) {
  for (int lv = 0; lv <= state->maxlevel; ++lv) {
    const uint32_t count = state->changed_old_link_counts[lv];
    const uint32_t base = changed_old_link_level_offset(state, lv);
    for (uint32_t idx = blockIdx.x; idx < count; idx += gridDim.x) {
      const uint32_t vid = state->changed_old_links[base + idx];
      bitonic_sort(state, vid, lv);
    }
  }
}

__device__ void bitonic_sort_id_for_all_ll(GpuGraphState *state) {
  for (int vid = blockIdx.x; vid < state->cur_element_count; vid += gridDim.x) {
    for (int lv = 0; lv <= state->element_levels[vid]; lv++) {
      uint32_t *linkl = get_level_linklist(state, vid, lv);
      const int len = getListCount(linkl);
      if (len <= 1 || len == get_frozen_link_count(state, vid, lv)) continue;
      bitonic_sort(state, vid, lv);
    }
  }
}

__device__ void bitonic_sort_pq(Neighbor *pq, unsigned len, bool inc = 1) {
  if (len <= 1) return;
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned sort_len = next_power_of_two(len);
  for (unsigned stride = 1; stride < sort_len; stride <<= 1) {
    for (unsigned step = stride; step > 0; step >>= 1) {
      for (unsigned k = tid; k < sort_len / 2; k += blockDim.x * blockDim.y) {
        unsigned a = 2 * step * (k / step);
        unsigned b = k % step;
        unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
        unsigned d = a + b + step;
        if (d < len && (inc ? pq[u].distance > pq[d].distance : pq[u].distance < pq[d].distance)) {
          swap(pq[u], pq[d]);
        }
      }
      __syncthreads();
    }
  }
}

__global__ void prepare_graph_kernel(GpuGraphState *state) {
  // compute the power of all the base vectors
  compute_power_kernel(state);
  // generate the random levels
  generate_random_levels_kernel(state);
  // copy the vector data to the half vector data
  copy_float_to_half_kernel(state);
}

cudaError_t launch_prepare_graph_kernel(GpuGraphState *state) {
  prepare_graph_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
  return cudaGetLastError();
}

__device__ void copy_float_to_half_kernel(GpuGraphState *state) {
  int wid = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = wid; i < state->max_elements * state->vector_dim; i += blockDim.x * gridDim.x) {
    state->half_vector_data[i] = __float2half(state->vector_data[i]);
  }
}

__device__ void reset_batch_state_kernel(GpuGraphState *state) {
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    state->old_vec_fetch_offset = 0;
  }
  for (int level = blockIdx.x * blockDim.x + threadIdx.x; level < MAX_HNSW_LEVEL; level += gridDim.x * blockDim.x) {
    state->changed_old_link_counts[level] = 0;
  }
}

__device__ void aggregate_on_level_kernel(GpuGraphState *state, int lv) {
  __shared__ uint32_t block_local_ids[LEVEL_SZ_THRES];
  __shared__ uint32_t block_match_count;
  __shared__ uint32_t block_write_base;

  if (threadIdx.x == 0) {
    block_match_count = 0;
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < state->cur_element_count; i += gridDim.x * blockDim.x) {
    if (state->element_levels[i] >= lv) {
      const uint32_t pos = atomicAdd(&block_match_count, 1U);
#ifndef NDEBUG
      if (pos >= LEVEL_SZ_THRES) {
        printf("Fatal: block %d aggregate overflow at level %d\n", blockIdx.x, lv);
        assert(false);
      }
#endif
      block_local_ids[pos] = i;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    block_write_base = atomicAdd(&state->old_vec_fetch_offset, block_match_count);
  }
  __syncthreads();

  for (uint32_t j = threadIdx.x; j < block_match_count; j += blockDim.x) {
    state->old_vector_fetch_index[block_write_base + j] = block_local_ids[j];
  }
}

// 1 block
__device__ void update_level_counts_kernel(GpuGraphState *state) {
  __shared__ uint32_t local_level_count[MAX_HNSW_LEVEL];
  for (int i = threadIdx.x; i < MAX_HNSW_LEVEL; i += blockDim.x) {
    local_level_count[i] = 0;
  }
  __syncthreads();

  int ed = min(state->cur_element_count + BATCHSZ_PER_NEW, state->max_elements);
  for (int i = state->cur_element_count + threadIdx.x; i < ed; i += blockDim.x) {
    uint32_t level = state->element_levels[i];
    atomicAdd(&local_level_count[level], 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = MAX_HNSW_LEVEL - 2; i >= 0; --i) {
      local_level_count[i] += local_level_count[i + 1];
    }
    for (int i = 0; i < MAX_HNSW_LEVEL; ++i) {
      state->level_counts[i] += local_level_count[i];
    }
  }
}

// Shared helper for new-new and new-old distance blocks.
// Computes row_vectors * col_vectors^T (inner products), then converts to squared L2:
// ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>.
__device__ void compute_dist_block_wmma(
    GpuGraphState *state,
    const half *row_vectors,
    const half *col_vectors,
    uint32_t row_count,
    uint32_t col_count,
    uint32_t output_col_offset,
    uint32_t row_id_base,
    const uint32_t *col_global_ids) {
  if (row_count == 0 || col_count == 0) return;

  namespace wmma = nvcuda::wmma;
  constexpr int kWmmaM = 16;
  constexpr int kWmmaN = 16;
  constexpr int kWmmaK = 16;
  const int warp_count = blockDim.x / warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const int tile_rows = row_count / kWmmaM;
  const int tile_cols = col_count / kWmmaN;
  const int tile_count = tile_rows * tile_cols;

  // WMMA for full 16x16 tiles.
  if (warp_id < warp_count) {
    for (int tile_idx = warp_id; tile_idx < tile_count; tile_idx += warp_count) {
      const int tile_row = tile_idx / tile_cols;
      const int tile_col = tile_idx % tile_cols;
      const int row0 = tile_row * kWmmaM;
      const int col0 = tile_col * kWmmaN;

      wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);

      for (int k = 0; k < state->vector_dim; k += kWmmaK) {
        wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_frag;
        const half *a_ptr = row_vectors + row0 * state->vector_dim + k;
        const half *b_ptr = col_vectors + col0 * state->vector_dim + k;
        wmma::load_matrix_sync(a_frag, a_ptr, state->vector_dim);
        wmma::load_matrix_sync(b_frag, b_ptr, state->vector_dim);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      float *output_ptr = state->news_dist + row0 * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + output_col_offset + col0;
      wmma::store_matrix_sync(output_ptr, c_frag, LEVEL_SZ_THRES + BATCHSZ_PER_NEW, wmma::mem_row_major);
    }
  }
  __syncthreads();

  // Scalar fallback for tails not covered by full WMMA tiles.
  const int wmma_rows = (row_count / 16) * 16;
  const int wmma_cols = (col_count / 16) * 16;
  for (uint32_t i = 0; i < row_count; ++i) {
    for (int j = threadIdx.x; j < col_count; j += blockDim.x) {
      if (i < wmma_rows && j < wmma_cols) continue;
      float ip = 0.0f;
      const int row_st = i * state->vector_dim;
      const int col_st = j * state->vector_dim;
      for (int k = 0; k < state->vector_dim; ++k) {
        ip += __half2float(row_vectors[row_st + k]) * __half2float(col_vectors[col_st + k]);
      }
      state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + output_col_offset + j] = ip;
    }
  }
  __syncthreads();

  // Convert inner products to squared L2 and write global ids to news_rank.
  for (uint32_t i = 0; i < row_count; ++i) {
    const uint32_t one = row_id_base + i;
    for (int j = threadIdx.x; j < col_count; j += blockDim.x) {
      const uint32_t another = col_global_ids[j];
      const int out_idx = i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + output_col_offset + j;
      float ip = state->news_dist[out_idx];
      state->news_dist[out_idx] = -2 * ip + state->vector_powers[one] + state->vector_powers[another];
      state->news_rank[out_idx] = another;
    }
  }
  __syncthreads();
}

__device__ void compute_new_new_dist_into_buffers(
    GpuGraphState *state,
    uint32_t batch_base,
    uint32_t new_count,
    float *dist_out,
    uint32_t dist_row_stride,
    uint32_t *rank_out,
    uint32_t rank_row_stride) {
  if (new_count == 0) return;

  half *new_vector_store = state->half_vector_data + batch_base * state->vector_dim;

  // compute new-new inner products using tensor core
  {
    namespace wmma = nvcuda::wmma;
    constexpr int kWmmaM = 16;
    constexpr int kWmmaN = 16;
    constexpr int kWmmaK = 16;
    const int warp_count = blockDim.x / warpSize;
    const int warp_id = threadIdx.x / warpSize;
    const int tile_cols = new_count / kWmmaN;
    const int tile_rows = new_count / kWmmaM;
    const int tile_count = tile_rows * tile_cols;

    if (warp_id < warp_count) {
      for (int tile_idx = warp_id; tile_idx < tile_count; tile_idx += warp_count) {
        const int tile_row = tile_idx / tile_cols;
        const int tile_col = tile_idx % tile_cols;
        const int new_row = tile_row * kWmmaM;
        const int new_col = tile_col * kWmmaN;

        wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k = 0; k < state->vector_dim; k += kWmmaK) {
          wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag;
          wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_frag;
          const half *a_ptr = new_vector_store + new_row * state->vector_dim + k;
          const half *b_ptr = new_vector_store + new_col * state->vector_dim + k;
          wmma::load_matrix_sync(a_frag, a_ptr, state->vector_dim);
          wmma::load_matrix_sync(b_frag, b_ptr, state->vector_dim);
          wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float *output_ptr = dist_out + new_row * dist_row_stride + new_col;
        wmma::store_matrix_sync(output_ptr, c_frag, dist_row_stride, wmma::mem_row_major);
      }
    }
  }
  __syncthreads();

  // handle tail rows/cols not covered by WMMA full tiles
  const int wmma_rows = (new_count / 16) * 16;
  const int wmma_cols = (new_count / 16) * 16;
  for (uint32_t i = 0; i < new_count; ++i) {
    for (int j = threadIdx.x; j < new_count; j += blockDim.x) {
      if (i < wmma_rows && j < wmma_cols) continue;
      float ip = 0.0f;
      const int row_st = i * state->vector_dim;
      const int col_st = j * state->vector_dim;
      for (int k = 0; k < state->vector_dim; ++k) {
        ip += __half2float(new_vector_store[row_st + k]) * __half2float(new_vector_store[col_st + k]);
      }
      dist_out[i * dist_row_stride + j] = ip;
    }
  }
  __syncthreads();

  for (uint32_t i = 0; i < new_count; ++i) {
    const uint32_t one = batch_base + i;
    for (int j = threadIdx.x; j < new_count; j += blockDim.x) {
      const int out_idx = i * dist_row_stride + j;
      float ip = dist_out[out_idx];
      dist_out[out_idx] = -2 * ip + state->vector_powers[one];
      const uint32_t another = batch_base + j;
      dist_out[out_idx] += state->vector_powers[another];
      rank_out[i * rank_row_stride + j] = another;
    }
  }
}

__device__ void load_precomputed_new_new_dist(GpuGraphState *state) {
  const uint32_t batch_id = state->cur_element_count / BATCHSZ_PER_NEW;
  const uint32_t new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  if (new_count == 0) return;

  const size_t batch_offset = batch_id * BATCHSZ_PER_NEW * BATCHSZ_PER_NEW;
  const float *src_dist = state->precomputed_new_new_dist + batch_offset;
  const uint32_t *src_rank = state->precomputed_new_new_rank + batch_offset;
  const int dst_row_stride = LEVEL_SZ_THRES + BATCHSZ_PER_NEW;
  const int total = new_count * new_count;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
    const int i = idx / new_count;
    const int j = idx % new_count;
    const int dst = i * dst_row_stride + LEVEL_SZ_THRES + j;
    const int src = i * BATCHSZ_PER_NEW + j;
    state->news_dist[dst] = src_dist[src];
    state->news_rank[dst] = src_rank[src];
  }
}

__global__ void precompute_new_new_dist_kernel(GpuGraphState *state) {
  const uint32_t batch_id = blockIdx.x;
  const uint32_t batch_base = batch_id * BATCHSZ_PER_NEW;
  if (batch_base >= state->max_elements) return;

  const uint32_t new_count = min(BATCHSZ_PER_NEW, state->max_elements - batch_base);
  const size_t batch_offset = batch_id * BATCHSZ_PER_NEW * BATCHSZ_PER_NEW;
  float *dist_out = state->precomputed_new_new_dist + batch_offset;
  uint32_t *rank_out = state->precomputed_new_new_rank + batch_offset;
  compute_new_new_dist_into_buffers(state, batch_base, new_count, dist_out, BATCHSZ_PER_NEW, rank_out, BATCHSZ_PER_NEW);
}

cudaError_t launch_precompute_new_new_dist_kernel(GpuGraphState *state, uint32_t max_elements) {
  const uint32_t num_batches = (max_elements + BATCHSZ_PER_NEW - 1) / BATCHSZ_PER_NEW;
  if (num_batches == 0) return cudaSuccess;
  precompute_new_new_dist_kernel<<<num_batches, BLOCK_DIM>>>(state);
  return cudaGetLastError();
}

// 1 block for 1 new vector
__device__ void connect_new_to_old_at_upper_kernel(GpuGraphState *state, int startup_lvl) {
  __shared__ uint32_t prev_neigh_rank;
  __shared__ uint32_t prev_neigh_id;
  __shared__ volatile uint32_t curr_neigh_cnt;
  __shared__ uint32_t neigh_rank[MAX_M0];
  __shared__ unsigned char pruned_mask[LEVEL_SZ_THRES + BATCHSZ_PER_NEW];
  __shared__ bool can_continue;
  // connect new-to-old edges from startup level
  int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  for (int bid = blockIdx.x; bid < new_count; bid += gridDim.x) {
    int vid = state->cur_element_count + bid;
    auto ranked_cand = state->news_rank + (bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW));
    auto ranked_dist = state->news_dist + (bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW));

    // prune candidates
    for (int lv = startup_lvl; lv <= state->element_levels[vid]; ++lv) {
      for (int i = threadIdx.x; i < state->old_vec_fetch_offset; i += blockDim.x) {
        if (state->element_levels[ranked_cand[i]] < lv) {
          pruned_mask[i] = 1;
        } else {
          pruned_mask[i] = 0;
        }
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        curr_neigh_cnt = 0;
        for (int i = 0; i < state->old_vec_fetch_offset; i++) {
          if (pruned_mask[i] == 0) {
            prev_neigh_rank = i;
            prev_neigh_id = ranked_cand[i];
            neigh_rank[0] = prev_neigh_rank;
            curr_neigh_cnt = 1;
            pruned_mask[i] = 1;
            break;
          }
        }
      }
      __syncthreads();
      int M = lv ? state->M : state->maxM0;

      while (curr_neigh_cnt < M) {
        // Start to treat the block as 2d, x for dimensions, y for candidates
        int tx = threadIdx.x % warpSize;  // compute distance along this dimension
        int ty = threadIdx.x / warpSize;  // compute for different candidates
        int nrow = blockDim.x / warpSize;

        if (threadIdx.x == 0) {
          can_continue = 0;
        }
        __syncthreads();
        const float *prev_vec = state->vector_data + prev_neigh_id * state->vector_dim;
        for (int i = ty; i < state->old_vec_fetch_offset; i += nrow) {
          if (i <= prev_neigh_rank) continue;
          if (pruned_mask[i] == 0) {
            can_continue = 1;
            float dist = 0.0f;
            int cand = ranked_cand[i];
            const float *cand_vec = state->vector_data + cand * state->vector_dim;
            // TODO: warp-level sync
            for (int j = tx; j < state->vector_dim; j += warpSize) {
              float diff = cand_vec[j] - prev_vec[j];
              dist += diff * diff;
            }
            for (int lane = warpSize / 2; lane > 0; lane /= 2) {
              dist += __shfl_down_sync(0xffffffff, dist, lane);
            }
            if (tx == 0 && dist < ranked_dist[i]) {
              pruned_mask[i] = 1;
            }
          }
        }
        __syncthreads();
        if (!can_continue) {
          break;
        }
        // Re-treat as 1d, adding the first survived edge.
        if (threadIdx.x == 0) {
          for (int i = prev_neigh_rank + 1; i < state->old_vec_fetch_offset; i++) {
            if (pruned_mask[i] == 0) {
              prev_neigh_rank = i;
              prev_neigh_id = ranked_cand[i];
              neigh_rank[curr_neigh_cnt] = i;
              curr_neigh_cnt++;
              pruned_mask[i] = 1;
              break;
            }
          }
        }
        __syncthreads();
      }

      // Still treat as 1d block, adding all the selected edges based on the result.
      uint32_t *linkl = get_level_linklist(state, vid, lv);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = get_level_linklist_dist(state, vid, lv);
      for (int i = threadIdx.x; i < curr_neigh_cnt; i += blockDim.x) {
        const uint32_t cand = ranked_cand[neigh_rank[i]];
#ifndef NDEBUG
        if (state->element_levels[cand] < lv) {
          printf("Fatal: connect_new_to_old selected off-level candidate %u at level %d for new %u\n", cand, lv, vid);
          assert(false);
        }
#endif
        int pos = atomicAdd((uint32_t *)linkl, 1);
#ifndef NDEBUG
        const uint32_t capacity = lv == 0 ? state->maxM0 + BATCHSZ_PER_NEW : state->M + BATCHSZ_PER_NEW;
        if (pos >= capacity) {
          printf("Fatal: new node link list out of bound at level %d for node %u\n", lv, vid);
          assert(false);
        }
#endif
        datal[pos] = cand;
        distl[pos] = ranked_dist[neigh_rank[i]];
        if (cand >= state->cur_element_count) continue;
        uint32_t *other_linkl = get_level_linklist(state, cand, lv);
        auto other_datal = (uint32_t *)(other_linkl) + 1;
        float *other_distl = get_level_linklist_dist(state, cand, lv);
        int other_pos = atomicAdd((uint32_t *)other_linkl, 1);
        record_changed_old_link(state, cand, lv);
#ifndef NDEBUG
        if (other_pos >= capacity) {
          printf("Fatal: reverse link list out of bound at level %d for node %u\n", lv, cand);
          assert(false);
        }
#endif
        other_datal[other_pos] = vid;
        other_distl[other_pos] = ranked_dist[neigh_rank[i]];
      }
    }
  }
}

// 1 block for 1 new vector
__device__ void combine_prune_for_new_kernel(GpuGraphState *state) {
  __shared__ uint32_t prev_neigh_rank;
  __shared__ uint32_t prev_neigh_id;
  __shared__ volatile uint32_t curr_neigh_cnt;
  __shared__ uint32_t neigh_rank[BATCHSZ_PER_NEW];
  __shared__ unsigned char pruned_mask[LEVEL_SZ_THRES + BATCHSZ_PER_NEW];
  __shared__ bool can_continue;
  __shared__ uint32_t shared_sz;
  // connect new-to-old and new-to-new edges from startup level
  int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  for (int bid = blockIdx.x; bid < new_count; bid += gridDim.x) {
    const int vid = state->cur_element_count + bid;
    auto ranked_cand = state->news_rank + (bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW));
    auto ranked_dist = state->news_dist + (bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW));

    for (int lv = 0; lv <= state->element_levels[vid]; ++lv) {
      // combine old neighbors and new vectors
      uint32_t *linkl = get_level_linklist(state, vid, lv);
      uint32_t sz = getListCount(linkl);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = get_level_linklist_dist(state, vid, lv);
      if (threadIdx.x == 0) {
        shared_sz = sz;
      }
      for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        int other = datal[i];
        float dist = distl[i];
        ranked_dist[i] = dist;
        ranked_cand[i] = other;
      }
      __syncthreads();
      static_assert(LEVEL_SZ_THRES - MAX_M0 >= BATCHSZ_PER_NEW);  // Avoid overwriting when moving data.
      for (int i = threadIdx.x; i < new_count; i += blockDim.x) {
        if (state->element_levels[state->cur_element_count + i] >= lv) {
          uint32_t pos = atomicAdd(&shared_sz, 1);
          ranked_dist[pos] = ranked_dist[i + LEVEL_SZ_THRES];
          ranked_cand[pos] = ranked_cand[i + LEVEL_SZ_THRES];
        }
      }
      __syncthreads();
      sz = shared_sz;

      int M = lv ? state->M : state->maxM0;
      // The branch that there is no need of pruning.
      if (sz < M + 1) {  // +1 for self-edge.
        for (int i = threadIdx.x; i < sz; i += blockDim.x) {
          datal[i] = ranked_cand[i + 1];
          distl[i] = ranked_dist[i + 1];
        }
        continue;
      }

      // sort the combined neighbors by distance
      int sortlen = next_power_of_two(sz);
      const unsigned tid = threadIdx.x;
      for (unsigned stride = 1; stride < sortlen; stride <<= 1) {
        for (unsigned step = stride; step > 0; step >>= 1) {
          for (unsigned k = tid; k < sortlen / 2; k += blockDim.x) {
            unsigned a = 2 * step * (k / step);
            unsigned b = k % step;
            unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
            unsigned d = a + b + step;
            if (d < sz && ranked_dist[u] > ranked_dist[d]) {
              swap(ranked_dist[u], ranked_dist[d]);
              swap(ranked_cand[u], ranked_cand[d]);
            }
          }
          __syncthreads();
        }
      }

      // prune all the candidates
      for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        pruned_mask[i] = 0;
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        pruned_mask[0] = 1;  // Avoid self-edge.
        curr_neigh_cnt = 0;
        for (int i = 1; i < sz; i++) {
          if (pruned_mask[i] == 0) {
            prev_neigh_rank = i;
            prev_neigh_id = ranked_cand[i];
            neigh_rank[0] = prev_neigh_rank;
            curr_neigh_cnt = 1;
            pruned_mask[i] = 1;
            break;
          }
        }
      }
      __syncthreads();
      if (curr_neigh_cnt == 0) {
        if (threadIdx.x == 0) {
          setListCount(linkl, 0);
        }
        __syncthreads();
        continue;
      }

      while (curr_neigh_cnt < M) {
        // Start to treat the block as 2d, x for dimensions, y for candidates
        int tx = threadIdx.x % warpSize;  // compute distance along this dimension
        int ty = threadIdx.x / warpSize;  // compute for different candidates
        int nrow = blockDim.x / warpSize;

        if (threadIdx.x == 0) {
          can_continue = 0;
        }
        __syncthreads();
        const float *prev_vec = state->vector_data + prev_neigh_id * state->vector_dim;
        for (int i = ty; i < sz; i += nrow) {
          if (i <= prev_neigh_rank) continue;
          if (pruned_mask[i] == 0) {
            can_continue = 1;
            float dist = 0.0f;
            int cand = ranked_cand[i];
            const float *cand_vec = state->vector_data + cand * state->vector_dim;
            for (int j = tx; j < state->vector_dim; j += warpSize) {
              float diff = cand_vec[j] - prev_vec[j];
              dist += diff * diff;
            }
            for (int lane = warpSize / 2; lane > 0; lane /= 2) {
              dist += __shfl_down_sync(0xffffffff, dist, lane);
            }
            if (tx == 0 && dist < ranked_dist[i]) {
              pruned_mask[i] = 1;
            }
          }
        }
        __syncthreads();
        if (!can_continue) {
          break;
        }
        // Re-treat as 1d, adding the first survived edge.
        if (threadIdx.x == 0) {
          for (int i = prev_neigh_rank + 1; i < sz; i++) {
            if (pruned_mask[i] == 0) {
              prev_neigh_rank = i;
              prev_neigh_id = ranked_cand[i];
              neigh_rank[curr_neigh_cnt] = i;
              curr_neigh_cnt++;
              pruned_mask[i] = 1;
              break;
            }
          }
        }
        __syncthreads();
      }

      // Still treat as 1d block, adding all the selected edges based on the result.
      for (int i = threadIdx.x; i < curr_neigh_cnt; i += blockDim.x) {
        datal[i] = ranked_cand[neigh_rank[i]];
        distl[i] = ranked_dist[neigh_rank[i]];
      }
      if (threadIdx.x == 0) {
        setListCount(linkl, curr_neigh_cnt);
      }
      __syncthreads();
    }
  }
}

// 1 block per new vector; reverse edges for levels below startup_level (deferred from search).
__device__ void add_reverse_edges_at_lower_kernel(GpuGraphState *state, int startup_level) {
  if (startup_level <= 0) {
    return;
  }
  const int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  for (int bid = blockIdx.x; bid < new_count; bid += gridDim.x) {
    const int vid = state->cur_element_count + bid;
    const int max_lv = min(startup_level - 1, state->element_levels[vid]);
    for (int lv = 0; lv <= max_lv; ++lv) {
      uint32_t *linkl = get_level_linklist(state, vid, lv);
      const int sz = getListCount(linkl);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = get_level_linklist_dist(state, vid, lv);
      for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        const uint32_t other = datal[i];
        if (other >= state->cur_element_count) {
          continue;
        }
        uint32_t *other_linkl = get_level_linklist(state, other, lv);
        uint32_t *other_datal = (uint32_t *)(other_linkl) + 1;
        float *other_distl = get_level_linklist_dist(state, other, lv);
        const uint32_t pos = atomicAdd((uint32_t *)(other_linkl), 1);
        record_changed_old_link(state, other, lv);
#ifndef NDEBUG
        const uint32_t capacity = lv == 0 ? state->maxM0 + BATCHSZ_PER_NEW : state->M + BATCHSZ_PER_NEW;
        if (pos >= capacity) {
          printf("Fatal: reverse link list out of bound at level %d for node %u\n", lv, other);
          assert(false);
        }
#endif
        other_datal[pos] = vid;
        other_distl[pos] = distl[i];
      }
      __syncthreads();
    }
  }
}

__device__ void snapshot_frozen_link_counts_kernel(GpuGraphState *state, int startup_level) {
  if (startup_level < 0) return;
  startup_level = min(startup_level - 1, state->maxlevel);
  const uint32_t new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  const uint32_t node_count = min(state->max_elements, state->cur_element_count + new_count);

  for (int lv = 0; lv <= startup_level; ++lv) {
    for (uint32_t node_id = threadIdx.x + blockIdx.x * blockDim.x; node_id < node_count;
         node_id += blockDim.x * gridDim.x) {
      uint32_t count = 0;
      if (state->element_levels[node_id] >= lv) {
        uint32_t *linkl = get_level_linklist(state, node_id, lv);
        count = getListCount(linkl);
#ifndef NDEBUG
        uint32_t *datal = (uint32_t *)(linkl + 1);
        for (uint32_t i = 0; i < count; ++i) {
          const uint32_t cand = datal[i];
          if (state->element_levels[cand] < lv) {
            printf("Fatal: snapshot found off-level candidate %u at level %d from node %u\n", cand, lv, node_id);
            assert(false);
          }
        }
#endif
      }
      state->frozen_link_counts[frozen_link_count_offset(state, node_id, lv)] = count;
    }
  }
}

__global__ void build_reset_batch_state_kernel(GpuGraphState *state) { reset_batch_state_kernel(state); }

__global__ void build_aggregate_on_level_kernel(GpuGraphState *state) {
  const int startup_level = compute_startup_level(state);
  uint64_t phase_t0 = build_phase_begin(state);
  // Aggregate the old vectors available on the startup level.
  aggregate_on_level_kernel(state, startup_level);
  build_phase_record(state, kBuildPhaseAggregate, phase_t0);
}

__global__ void build_dist_old_new_and_load_new_new_kernel(GpuGraphState *state) {
  uint64_t phase_t0 = build_phase_begin(state);
  // Compute the distances between the new vectors and the old vectors.
  compute_dist_with_old_kernel(state);
  build_phase_record(state, kBuildPhaseDistOldNew, phase_t0);

  phase_t0 = build_phase_begin(state);
  // Load the precomputed distances between the new vectors for this batch.
  load_precomputed_new_new_dist(state);
  build_phase_record(state, kBuildPhaseLoadNewNew, phase_t0);
}

__global__ void build_sort_and_connect_upper_kernel(GpuGraphState *state) {
  const int startup_level = compute_startup_level(state);
  uint64_t phase_t0 = build_phase_begin(state);
  // For the new vectors, sort the aggregated old vectors by distance
  bitonic_sort_id_by_dis(state);
  build_phase_record(state, kBuildPhaseSortOldByDist, phase_t0);

  phase_t0 = build_phase_begin(state);
  // Connect the new vectors to the old vectors at upper levels (>= startup_level).
  connect_new_to_old_at_upper_kernel(state, startup_level);
  build_phase_record(state, kBuildPhaseConnectUpper, phase_t0);
}

__global__ void build_snapshot_frozen_kernel(GpuGraphState *state) {
  const int startup_level = compute_startup_level(state);
  uint64_t phase_t0 = build_phase_begin(state);
  // Freeze the link counts at lower levels (< startup_level)
  // so that new vectors won't process the added reverse edges in current batch.
  snapshot_frozen_link_counts_kernel(state, startup_level);
  build_phase_record(state, kBuildPhaseSnapshotFrozen, phase_t0);
}

__global__ void build_search_knn_lower_kernel(GpuGraphState *state) {
  const int startup_level = compute_startup_level(state);
  uint64_t phase_t0 = build_phase_begin(state);
  search_knn_at_lower_kernel(state, startup_level);
  build_phase_record(state, kBuildPhaseSearchLower, phase_t0);
}

__global__ void build_combine_prune_new_kernel(GpuGraphState *state) {
  uint64_t phase_t0 = build_phase_begin(state);
  combine_prune_for_new_kernel(state);
  build_phase_record(state, kBuildPhaseCombinePruneNew, phase_t0);
}

__global__ void build_add_reverse_lower_kernel(GpuGraphState *state) {
  const int startup_level = compute_startup_level(state);
  add_reverse_edges_at_lower_kernel(state, startup_level);
}

__global__ void build_sort_prune_old_kernel(GpuGraphState *state) {
  uint64_t phase_t0 = build_phase_begin(state);
  // Sort and prune only old-node lists that received reverse edges in this batch.
  bitonic_sort_id_for_ll(state);
  // Update frozen link counts for the levels that received reverse edges.
  prune_for_old_kernel(state);
  build_phase_record(state, kBuildPhaseSortPruneOld, phase_t0);
}

__global__ void build_update_batch_kernel(GpuGraphState *state) {
  uint64_t phase_t0 = build_phase_begin(state);
  const int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  if (blockIdx.x == 0) {
    update_level_counts_kernel(state);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    state->cur_element_count += new_count;
#ifdef PROFILE_BUILD_PHASES
    if (state->profile_build_phases) {
      state->build_batch_count += 1ULL;
    }
#endif
#ifndef NDEBUG
    if (state->cur_element_count % 1024 == 0) {
      printf("cur_element_count=%d\n", state->cur_element_count);
    }
#endif
  }
  build_phase_record(state, kBuildPhaseUpdateBatch, phase_t0);
}

cudaError_t launch_build_graph_kernel(GpuGraphState *state) {
  uint32_t max_elements = 0;
  cudaMemcpy(&max_elements, &state->max_elements, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  for (uint32_t i = 0; i < max_elements; i += BATCHSZ_PER_NEW) {
    build_reset_batch_state_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    build_aggregate_on_level_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    build_dist_old_new_and_load_new_new_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    build_sort_and_connect_upper_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    build_snapshot_frozen_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    build_search_knn_lower_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    build_combine_prune_new_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    build_add_reverse_lower_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    build_sort_prune_old_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    build_update_batch_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
  }
  return cudaGetLastError();
}

// 1d grid, 1d block
__device__ void compute_power_kernel(GpuGraphState *state) {
  int bid = blockIdx.x;
  if (bid >= state->max_elements) {
    return;
  }
  __shared__ float cache[MAX_DIM];
  while (bid < state->max_elements) {
    int vector_start = bid * state->vector_dim;
    int vector_end = vector_start + state->vector_dim;
    int wid = vector_start + threadIdx.x;
    int cid = threadIdx.x;
    float temp = 0.0f;
    // TODO: Rewrite this to intra-warp reduction.
    while (wid < vector_end) {
      temp += state->vector_data[wid] * state->vector_data[wid];
      wid += blockDim.x;
    }
    cache[cid] = temp;
    __syncthreads();

    int active_threads = state->vector_dim;
    while (active_threads > 1) {
      int half = active_threads / 2;
      if (cid < half) {
        cache[cid] += cache[cid + half];
      }
      if ((active_threads & 1) && cid == 0) {
        cache[0] += cache[active_threads - 1];
      }
      active_threads = half;
      __syncthreads();
    }
    if (cid == 0) {
      state->vector_powers[bid] = cache[0];
    }
    bid += gridDim.x;
  }
}

// 1d grid, 1d block
__device__ void generate_random_levels_kernel(GpuGraphState *state) {
  int local_maxlevel = 0;

  int wid = threadIdx.x + blockDim.x * blockIdx.x;
  while (wid < state->max_elements) {
    curandStatePhilox4_32_10_t rng_state;
    curand_init(kRandomLevelSeed, wid, 0, &rng_state);
    float sample = fmaxf(curand_uniform(&rng_state), 1.0e-7f);

    int32_t level = -logf(sample) * state->mult;
    level = min(level, MAX_HNSW_LEVEL - 1);
    state->element_levels[wid] = level;

    if (level > local_maxlevel) {
      local_maxlevel = level;
    }
    wid += blockDim.x * gridDim.x;
  }
  atomicMax(&state->maxlevel, local_maxlevel);
}

// 1d grid, 1d block, 1 block per batch of old vectors
__device__ void compute_dist_with_old_kernel(GpuGraphState *state) {
  half *new_vector_store = state->half_vector_data + state->cur_element_count * state->vector_dim;
  half *old_vector_store = state->old_vector_store;

  for (int bid = blockIdx.x; bid * BATCHSZ_PER_OLD < state->old_vec_fetch_offset; bid += gridDim.x) {
    // load old vectors
    const int oid = bid * BATCHSZ_PER_OLD;
    const uint32_t new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
    const uint32_t old_count = min(BATCHSZ_PER_OLD, state->old_vec_fetch_offset - oid);
    if (old_count == 0 || new_count == 0) {
      continue;
    }
    for (uint32_t i = 0; i < old_count; ++i) {
      const uint32_t vector_st = state->old_vector_fetch_index[i + oid] * state->vector_dim;
      const uint32_t old_vector_store_st = (i + oid) * state->vector_dim;
      for (int j = threadIdx.x; j < state->vector_dim; j += blockDim.x) {
        old_vector_store[old_vector_store_st + j] = state->half_vector_data[vector_st + j];
      }
    }

    __syncthreads();

    // compute new-old inner products using tensor core
    // vector_dim is padded to a multiple of 16 on H2D (move_to_gpu).
    {
      namespace wmma = nvcuda::wmma;
      constexpr int kWmmaM = 16;
      constexpr int kWmmaN = 16;
      constexpr int kWmmaK = 16;
      const int warp_count = blockDim.x / warpSize;
      const int warp_id = threadIdx.x / warpSize;
      const int tile_rows = new_count / kWmmaM;
      const int tile_cols = old_count / kWmmaN;
      const int tile_count = tile_rows * tile_cols;

      if (warp_id < warp_count) {  // TODO: under-utilized due to the (warp count) * warpSize < blockDim.x
        for (int tile_idx = warp_id; tile_idx < tile_count; tile_idx += warp_count) {
          const int tile_row = tile_idx / tile_cols;
          const int tile_col = tile_idx % tile_cols;
          const int new_row = tile_row * kWmmaM;
          const int old_col = tile_col * kWmmaN;

          wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
          wmma::fill_fragment(c_frag, 0.0f);

          for (int k = 0; k < state->vector_dim; k += kWmmaK) {
            wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_frag;
            const half *a_ptr = new_vector_store + new_row * state->vector_dim + k;
            const half *b_ptr = old_vector_store + (oid + old_col) * state->vector_dim + k;
            wmma::load_matrix_sync(a_frag, a_ptr, state->vector_dim);
            wmma::load_matrix_sync(b_frag, b_ptr, state->vector_dim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
          }

          float *output_ptr = state->news_dist + new_row * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + oid + old_col;
          wmma::store_matrix_sync(output_ptr, c_frag, LEVEL_SZ_THRES + BATCHSZ_PER_NEW, wmma::mem_row_major);
        }
      }
    }

    // handle tail rows/cols not covered by WMMA full tiles
    const int wmma_rows = (new_count / 16) * 16;
    const int wmma_cols = (old_count / 16) * 16;
    if (new_count % 16 != 0 || old_count % 16 != 0) {  // only the last batch may not be covered by WMMA tiling
      for (uint32_t i = 0; i < new_count; ++i) {
        for (int local_j = threadIdx.x; local_j < old_count; local_j += blockDim.x) {
          if (i < wmma_rows && local_j < wmma_cols) continue;
          float ip = 0.0f;
          const int new_st = i * state->vector_dim;
          const int old_st = (oid + local_j) * state->vector_dim;
          for (int k = 0; k < state->vector_dim; ++k) {
            ip += __half2float(new_vector_store[new_st + k]) * __half2float(old_vector_store[old_st + k]);
          }
          state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + oid + local_j] = ip;
        }
      }
    }
    __syncthreads();

    // convert inner product to L2 distance for this old batch range
    for (uint32_t i = 0; i < new_count; ++i) {
      const uint32_t one = state->cur_element_count + i;
      for (int j = oid + threadIdx.x; j < oid + old_count; j += blockDim.x) {
        float ip = state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + j];
        state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + j] = -2 * ip;
        state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + j] += state->vector_powers[one];
        const uint32_t another = state->old_vector_fetch_index[j];
        state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + j] += state->vector_powers[another];
        state->news_rank[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + j] = another;
      }
    }
    __syncthreads();
  }
}

// 1 block for 1 old vector
// assume that neighbor list has been sorted
__device__ void prune_for_old_kernel(GpuGraphState *state) {
  __shared__ uint32_t prev_neigh_rank;
  __shared__ uint32_t prev_neigh_id;
  __shared__ volatile uint32_t curr_neigh_cnt;
  __shared__ unsigned char pruned_mask[LEVEL_SZ_THRES + BATCHSZ_PER_NEW];
  __shared__ bool can_continue;

  for (int lv = 0; lv <= state->maxlevel; ++lv) {
    const uint32_t count = state->changed_old_link_counts[lv];
    const uint32_t base = changed_old_link_level_offset(state, lv);
    for (uint32_t idx = blockIdx.x; idx < count; idx += gridDim.x) {
      const uint32_t vid = state->changed_old_links[base + idx];
      int M = lv ? state->M : state->maxM0;

      uint32_t *linkl = get_level_linklist(state, vid, lv);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = get_level_linklist_dist(state, vid, lv);
      int sz = getListCount(linkl);

      if (sz > M) {
        if (threadIdx.x == 0) {
          prev_neigh_rank = 0;
          prev_neigh_id = datal[0];
          datal[0] = prev_neigh_id;
          curr_neigh_cnt = 1;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < sz; i += blockDim.x) {
          pruned_mask[i] = 0;
        }
        __syncthreads();

        while (curr_neigh_cnt < M) {
          if (threadIdx.x == 0) {
            can_continue = 0;
          }
          __syncthreads();
          const int tx = threadIdx.x % warpSize;  // compute distance along this dimension
          const int ty = threadIdx.x / warpSize;  // compute for different candidates
          const int nrow = blockDim.x / warpSize;
          const float *prev_vec = state->vector_data + prev_neigh_id * state->vector_dim;

          for (int i = ty; i < sz; i += nrow) {
            if (i <= prev_neigh_rank) continue;
            if (pruned_mask[i] == 0) {
              can_continue = 1;
              float dist = 0.0f;
              const int cand = datal[i];
              const float *cand_vec = state->vector_data + cand * state->vector_dim;
              for (int j = tx; j < state->vector_dim; j += warpSize) {
                float diff = cand_vec[j] - prev_vec[j];
                dist += diff * diff;
              }
              for (int lane = warpSize / 2; lane > 0; lane /= 2) {
                dist += __shfl_down_sync(0xffffffff, dist, lane);
              }
              if (tx == 0 && dist < distl[i]) {
                pruned_mask[i] = 1;
              }
            }
          }
          __syncthreads();
          if (!can_continue) {
            break;
          }
          if (threadIdx.x == 0) {
            for (int i = prev_neigh_rank + 1; i < sz; i++) {
              if (pruned_mask[i] == 0) {
                prev_neigh_rank = i;
                prev_neigh_id = datal[i];
                distl[curr_neigh_cnt] = distl[i];
                datal[curr_neigh_cnt] = datal[i];
                curr_neigh_cnt++;
                pruned_mask[i] = 1;
                break;
              }
            }
          }
          __syncthreads();
        }

        if (threadIdx.x == 0) {
          setListCount(linkl, curr_neigh_cnt);
        }
      }
      if (threadIdx.x == 0) {
        state->frozen_link_counts[frozen_link_count_offset(state, vid, lv)] = getListCount(linkl);
      }
      __syncthreads();
    }
  }
}

// 1 block for 1 new vector
// threads for distance computation
__device__ void search_knn_at_lower_kernel(GpuGraphState *state, int startup_lv) {
  __shared__ uint32_t *visited;
  __shared__ uint32_t visited_tag;
  __shared__ Neighbor candq[CANDQ_SZ];
  __shared__ int candq_sz;
  __shared__ Neighbor topq[TOPQ_SZ];
  __shared__ int topq_sz;
  __shared__ float topq_max;
  __shared__ int changed;
  __shared__ uint32_t curr_obj_shared;
  __shared__ int curr_dist_bits_shared;
  __shared__ Neighbor warp_staging[SEARCH_WARP_COUNT][WARP_STAGING_CAP];
  __shared__ int warp_staging_sz[SEARCH_WARP_COUNT];
  __shared__ float warp_best_dist[SEARCH_WARP_COUNT];
  __shared__ uint32_t warp_best_cand[SEARCH_WARP_COUNT];
  __shared__ Neighbor popped;
  __shared__ int search_continue;
  const int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  if (threadIdx.x == 0) {
    visited = state->visited + blockIdx.x * state->max_elements;
    visited_tag = 0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < state->max_elements; i += blockDim.x) {
    visited[i] = -1;
  }
  __syncthreads();
  for (int bid = blockIdx.x; bid < new_count; bid += gridDim.x) {
    const int vid = state->cur_element_count + bid;
    const int lvl = state->element_levels[vid];
    const int tx = threadIdx.x % warpSize;
    const int ty = threadIdx.x / warpSize;
    const int nrow = blockDim.x / warpSize;

    float *query_vec = state->vector_data + vid * state->vector_dim;
    // Find the entry point
    if (threadIdx.x == 0) {
      auto ranks = state->news_rank + bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
      auto dists = state->news_dist + bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
      curr_obj_shared = ranks[0];
      curr_dist_bits_shared = __float_as_int(dists[0]);
    }
    __syncthreads();

    int lv = startup_lv - 1;
    while (lv > lvl) {
      while (1) {
        if (threadIdx.x == 0) {
          changed = 0;
        }
        __syncthreads();

        const int size = get_frozen_link_count(state, curr_obj_shared, lv);
        uint32_t *linkl = size > 0 ? get_level_linklist(state, curr_obj_shared, lv) : nullptr;
        uint32_t *datal = size > 0 ? (uint32_t *)(linkl + 1) : nullptr;

        float local_best;
        uint32_t local_cand;
        if (tx == 0) {
          local_best = __int_as_float(curr_dist_bits_shared);
          local_cand = curr_obj_shared;
        }
        for (int i = ty; i < size; i += nrow) {
          uint32_t cand = datal[i];
          float dist = 0.0f;
          for (int j = tx; j < state->vector_dim; j += warpSize) {
            const float diff = state->vector_data[cand * state->vector_dim + j] - query_vec[j];
            dist += diff * diff;
          }
          for (int lane = warpSize / 2; lane > 0; lane /= 2) {
            dist += __shfl_down_sync(0xffffffff, dist, lane);
          }
          if (tx == 0 && dist < local_best) {
            local_best = dist;
            local_cand = cand;
          }
        }
        if (tx == 0) {
          warp_best_dist[ty] = local_best;
          warp_best_cand[ty] = local_cand;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          float curr_dist = __int_as_float(curr_dist_bits_shared);
          for (int w = 0; w < SEARCH_WARP_COUNT; ++w) {
            if (warp_best_dist[w] < curr_dist) {
              curr_dist = warp_best_dist[w];
              curr_obj_shared = warp_best_cand[w];
              changed = 1;
            }
          }
          curr_dist_bits_shared = __float_as_int(curr_dist);
        }
        __syncthreads();
        if (!changed) {
          break;
        }
      }
      lv--;
    }
    // Search and insert at current level.
    while (lv >= 0) {
      if (threadIdx.x == 0) {
        candq_sz = 0;
        float curr_dist = __int_as_float(curr_dist_bits_shared);
        uint32_t curr_obj = curr_obj_shared;
        MinPqPush(candq, &candq_sz, curr_dist, curr_obj, 0);
        topq_sz = 0;
        MaxPqPush(topq, &topq_sz, curr_dist, curr_obj, 1);
        topq_max = curr_dist;
        ++visited_tag;
      }
      __syncthreads();

      while (true) {
        if (threadIdx.x == 0) {
          search_continue =
              (candq_sz > 0) && !(topq_sz >= static_cast<int>(state->ef_construction) && candq[0].distance > topq_max);
          if (search_continue) {
            popped = candq[0];
            visited[popped.nodeid] = visited_tag;
            MinPqPop(candq, &candq_sz, &popped);
          }
        }
        __syncthreads();
        if (!search_continue) {
          break;
        }

        const int tx = threadIdx.x % warpSize;
        const int ty = threadIdx.x / warpSize;
        const int nrow = blockDim.x / warpSize;
        const uint32_t node = popped.nodeid;

        if (tx == 0) {
          warp_staging_sz[ty] = 0;
        }
        const int size = get_frozen_link_count(state, node, lv);
        uint32_t *linkl = size > 0 ? get_level_linklist(state, node, lv) : nullptr;
        uint32_t *datal = size > 0 ? (uint32_t *)(linkl + 1) : nullptr;
        for (int i = ty; i < size; i += nrow) {
          uint32_t cand = datal[i];
#ifndef NDEBUG
          if (state->element_levels[cand] < lv) {
            printf("Fatal: found off-level candidate %u at level %d from node %u\n", cand, lv, node);
            assert(false);
          }
#endif
          uint32_t prev_tag = 0;
          if (tx == 0) {
            prev_tag = atomicExch(&visited[cand], visited_tag);
          }
          prev_tag = __shfl_sync(0xffffffff, prev_tag, 0);
          if (prev_tag == visited_tag) {
            continue;
          }
          float dist = 0.0f;
          for (int j = tx; j < state->vector_dim; j += warpSize) {
            const float diff = state->vector_data[cand * state->vector_dim + j] - query_vec[j];
            dist += diff * diff;
          }
          for (int lane = warpSize / 2; lane > 0; lane /= 2) {
            dist += __shfl_down_sync(0xffffffff, dist, lane);
          }
          if (tx == 0) {
            const int pos = warp_staging_sz[ty];
            if (pos < WARP_STAGING_CAP) {
              warp_staging[ty][pos].distance = dist;
              warp_staging[ty][pos].nodeid = cand;
              warp_staging[ty][pos].checked = false;
              warp_staging_sz[ty] = pos + 1;
            }
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          merge_staged_neighbors_into_queues(
              candq, &candq_sz, topq, &topq_sz, &topq_max, warp_staging, warp_staging_sz, state->ef_construction);
        }
        __syncthreads();
      }

      uint32_t *linkl = get_level_linklist(state, vid, lv);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = get_level_linklist_dist(state, vid, lv);
      const int write_sz = min(topq_sz, TOPQ_SZ);
      for (int i = threadIdx.x; i < write_sz; i += blockDim.x) {
        datal[i] = topq[i].nodeid;
        distl[i] = topq[i].distance;
      }
      if (threadIdx.x == 0) {
        setListCount(linkl, write_sz);
        float entry_dist = 0.0f;
        find_closest_in_topq(topq, write_sz, &curr_obj_shared, &entry_dist);
        curr_dist_bits_shared = __float_as_int(entry_dist);
#ifndef NDEBUG
        const uint32_t capacity = lv == 0 ? state->maxM0 + BATCHSZ_PER_NEW : state->M + BATCHSZ_PER_NEW;
        if (write_sz > capacity) {
          printf("Fatal: staged search results exceed link list capacity\n");
          assert(false);
        }
#endif
      }
      __syncthreads();

      lv--;
      __syncthreads();
    }
  }
}

#ifdef PROFILE_BUILD_PHASES
#include <fmt/format.h>

void print_build_phase_profile(const uint64_t *cycles, uint64_t batch_count) {
  if (batch_count == 0 || cycles == nullptr) {
    fmt::print("build_graph_kernel phase profile: no batches recorded\n");
    return;
  }

  int clock_rate_khz = 0;
  cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
  if (clock_rate_khz <= 0) {
    fmt::print("build_graph_kernel phase profile: unable to read GPU clock rate\n");
    return;
  }

  const double cycles_per_ms = clock_rate_khz;
  double total_ms = 0.0;
  double phase_ms[kBuildPhaseCount]{};
  for (int phase = 0; phase < kBuildPhaseCount; ++phase) {
    phase_ms[phase] = cycles[phase] / cycles_per_ms;
    total_ms += phase_ms[phase];
  }

  fmt::print(
      "build_graph_kernel phase profile ({} batches, {:.3f} ms measured, {:.3f} ms per batch):\n",
      batch_count,
      total_ms,
      total_ms / batch_count);
  for (int phase = 0; phase < kBuildPhaseCount; ++phase) {
    const double pct = total_ms > 0.0 ? (phase_ms[phase] * 100.0 / total_ms) : 0.0;
    fmt::print(
        "  {:<22} {:>10.3f} ms  {:>5.1f}%  {:>8.3f} ms/batch\n",
        build_phase_name(static_cast<BuildPhase>(phase)),
        phase_ms[phase],
        pct,
        phase_ms[phase] / batch_count);
  }
}
#endif  // PROFILE_BUILD_PHASES
