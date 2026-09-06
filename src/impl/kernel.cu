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
}  // namespace

struct Neighbor {
  volatile float distance;
  volatile int nodeid;
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
                   sizeof(linklistsizeint) + (state->M + REVERSE_HEADROOM) * sizeof(uint32_t));
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
                   (state->maxM0 + REVERSE_HEADROOM) * sizeof(uint32_t));
}

__device__ uint32_t *get_level_linklist(GpuGraphState *state, uint32_t internal_id, int level) {
  return level == 0 ? get_linklist0(state, internal_id) : get_linklist(state, internal_id, level);
}

__device__ float *get_level_linklist_dist(GpuGraphState *state, uint32_t internal_id, int level) {
  return level == 0 ? get_linklist_dist0(state, internal_id) : get_linklist_dist(state, internal_id, level);
}

__device__ uint32_t getListCount(uint32_t *ptr) { return *((uint32_t *)ptr); }

__device__ void setListCount(uint32_t *ptr, uint32_t size) { *((tableint *)ptr) = size; }

__device__ uint32_t link_capacity_at_level(const GpuGraphState *state, int level) {
  return level == 0 ? state->maxM0 + REVERSE_HEADROOM : state->M + REVERSE_HEADROOM;
}

__device__ uint32_t frozen_link_count_offset(GpuGraphState *state, uint32_t internal_id, int level) {
  return level * state->max_elements + internal_id;
}

__device__ uint32_t get_frozen_link_count(GpuGraphState *state, uint32_t internal_id, int level) {
  if (level < 0 || level >= MAX_HNSW_LEVEL || internal_id >= state->max_elements ||
      state->element_levels[internal_id] < level) {
    return 0;
  }
  // Reverse edges are appended after search, so the live count is the frozen prefix.
  return getListCount(get_level_linklist(state, internal_id, level));
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

__device__ void copy_neighbor(Neighbor *dst, const Neighbor *src) {
  dst->distance = src->distance;
  dst->nodeid = src->nodeid;
}

__device__ void MaxPqPop(Neighbor *pq, volatile int *size) {
  if (*size == 0) return;
  (*size)--;
  float tail_dist = pq[*size].distance;
  int p = 0, r = 1;
  while (r < *size) {
    if (r < (*size) - 1 && gt(pq[r + 1].distance, pq[r].distance)) r++;
    if (ge(tail_dist, pq[r].distance)) break;
    copy_neighbor(&pq[p], &pq[r]);
    p = r;
    r = 2 * p + 1;
  }
  copy_neighbor(&pq[p], &pq[*size]);
}

__device__ void MinPqPop(Neighbor *pq, volatile int *size, Neighbor *tmp) {
  if (*size == 0) {
    printf("Fatal: MinPqPop: queue size == 0\n");
    assert(false);
  }
  (*size)--;
  copy_neighbor(tmp, &pq[0]);
  float tail_dist = pq[*size].distance;
  int p = 0, r = 1;
  while (r < *size) {
    if (r < (*size) - 1 && lt(pq[r + 1].distance, pq[r].distance)) r++;
    if (le(tail_dist, pq[r].distance)) break;
    copy_neighbor(&pq[p], &pq[r]);
    p = r;
    r = 2 * p + 1;
  }
  copy_neighbor(&pq[p], &pq[*size]);
}

__device__ void MaxPqPush(Neighbor *pq, volatile int *size, float dist, int nodeid, bool check) {
  int idx = *size;
  while (idx > 0) {
    int nidx = (idx + 1) / 2 - 1;
    if (ge(pq[nidx].distance, dist)) break;
    copy_neighbor(&pq[idx], &pq[nidx]);
    idx = nidx;
  }
  pq[idx].distance = dist;
  pq[idx].nodeid = nodeid;
  (*size)++;
}

__device__ void MinPqPush(Neighbor *pq, volatile int *size, float dist, int nodeid, bool check) {
  int idx = *size;
  while (idx > 0) {
    int nidx = (idx + 1) / 2 - 1;
    if (le(pq[nidx].distance, dist)) break;
    copy_neighbor(&pq[idx], &pq[nidx]);
    idx = nidx;
  }
  pq[idx].distance = dist;
  pq[idx].nodeid = nodeid;
  (*size)++;
}

__device__ void find_closest_in_queue(const Neighbor *queue, int sz, uint32_t *out_id, float *out_dist) {
  if (sz <= 0) {
    printf("Fatal: queue size <= 0\n");
    assert(false);
    return;
  }
  int best = 0;
  for (int i = 1; i < sz; ++i) {
    if (queue[i].distance < queue[best].distance) {
      best = i;
    }
  }
  *out_id = queue[best].nodeid;
  *out_dist = queue[best].distance;
}

__device__ void merge_staged_neighbors_into_queues(
    Neighbor *candq,
    volatile int *candq_sz,
    Neighbor *topq,
    volatile int *topq_sz,
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
        } else {
#ifndef NDEBUG
          printf("Fatal: candq overflow %d > %d\n", *candq_sz, CANDQ_SZ);
#endif
        }
        while (*topq_sz >= ef_construction) {
          MaxPqPop(topq, topq_sz);
        }
        MaxPqPush(topq, topq_sz, dist, nodeid, true);
        *topq_max = topq[0].distance;
      }
    }
  }
}

__device__ void bitonic_sort_id_by_dis(GpuGraphState *state, int startup_level) {
  __shared__ float warp_min_dist[SEARCH_WARP_COUNT];
  __shared__ unsigned warp_min_pos[SEARCH_WARP_COUNT];
  int len = state->old_vec_fetch_offset;
  float *distances = state->news_dist;
  unsigned *ids = state->news_rank;
  if (len <= 1) return;
  const unsigned sort_len = next_power_of_two(len);
  const int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  for (int bid = blockIdx.x; bid < new_count; bid += gridDim.x) {
    float *target = distances + bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
    unsigned *target_ids = ids + bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
    const unsigned tid = threadIdx.x;
    const uint32_t new_id = state->cur_element_count + bid;
    if (state->element_levels[new_id] >= static_cast<uint32_t>(startup_level)) {
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
      continue;
    }

    float best_dist = __int_as_float(0x7f800000);
    unsigned best_pos = ~0U;
    for (unsigned i = tid; i < static_cast<unsigned>(len); i += blockDim.x) {
      const float dist = target[i];
      if (dist < best_dist || (dist == best_dist && i < best_pos)) {
        best_dist = dist;
        best_pos = i;
      }
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_dist = __shfl_down_sync(0xffffffff, best_dist, offset);
      const unsigned other_pos = __shfl_down_sync(0xffffffff, best_pos, offset);
      if (other_dist < best_dist || (other_dist == best_dist && other_pos < best_pos)) {
        best_dist = other_dist;
        best_pos = other_pos;
      }
    }
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    if (lane == 0) {
      warp_min_dist[warp] = best_dist;
      warp_min_pos[warp] = best_pos;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      best_dist = warp_min_dist[0];
      best_pos = warp_min_pos[0];
      for (int i = 1; i < SEARCH_WARP_COUNT; ++i) {
        const float dist = warp_min_dist[i];
        const unsigned pos = warp_min_pos[i];
        if (dist < best_dist || (dist == best_dist && pos < best_pos)) {
          best_dist = dist;
          best_pos = pos;
        }
      }
      swap(target[0], target[best_pos]);
      swap(target_ids[0], target_ids[best_pos]);
    }
    __syncthreads();
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
  const size_t n = static_cast<size_t>(state->max_elements) * state->vector_dim;
  for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += blockDim.x * gridDim.x) {
    state->half_vector_data[i] = __float2half(state->vector_data[i]);
  }
}

__device__ void reset_batch_state_kernel(GpuGraphState *state) {
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    state->old_vec_fetch_offset = 0;
    state->next_bid = 0;
  }
  for (int level = blockIdx.x * blockDim.x + threadIdx.x; level < MAX_HNSW_LEVEL; level += gridDim.x * blockDim.x) {
    state->changed_old_link_counts[level] = 0;
    state->old_vector_level_delimiter[level] = 0;
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
    const uint32_t id = block_local_ids[j];
    const uint32_t level = state->element_levels[id];
    state->old_vector_fetch_index[block_write_base + j] = id;
    atomicAdd(&state->old_vector_level_delimiter[level], 1);
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

// Grid-wide A*A^T of the current new batch into news_dist columns [LEVEL_SZ_THRES, ...).
// Each warp owns its 16x16 tiles end-to-end (MMA + L2 epilogue), so no grid barrier is required.
__device__ void compute_new_new_dist_kernel(GpuGraphState *state) {
  const uint32_t batch_base = state->cur_element_count;
  const uint32_t new_count = min(BATCHSZ_PER_NEW, state->max_elements - batch_base);
  if (new_count == 0) return;

  const half *new_vector_store = state->half_vector_data + static_cast<size_t>(batch_base) * state->vector_dim;
  constexpr int kStride = LEVEL_SZ_THRES + BATCHSZ_PER_NEW;
  constexpr int kColOff = LEVEL_SZ_THRES;

  namespace wmma = nvcuda::wmma;
  constexpr int kWmmaM = 16;
  constexpr int kWmmaN = 16;
  constexpr int kWmmaK = 16;
  const int warps_per_block = blockDim.x / warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const int global_warp = blockIdx.x * warps_per_block + warp_id;
  const int total_warps = gridDim.x * warps_per_block;
  const int tile_cols = new_count / kWmmaN;
  const int tile_rows = new_count / kWmmaM;
  const int tile_count = tile_rows * tile_cols;
  __shared__ float ip_tile[BLOCK_DIM / 32][kWmmaM * kWmmaN];

  for (int tile_idx = global_warp; tile_idx < tile_count; tile_idx += total_warps) {
    const int new_row = (tile_idx / tile_cols) * kWmmaM;
    const int new_col = (tile_idx % tile_cols) * kWmmaN;

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

    wmma::store_matrix_sync(&ip_tile[warp_id][0], c_frag, kWmmaN, wmma::mem_row_major);
    __syncwarp();

    const int lane = threadIdx.x % warpSize;
    for (int e = lane; e < kWmmaM * kWmmaN; e += warpSize) {
      const int ti = e / kWmmaN;
      const int tj = e % kWmmaN;
      const uint32_t one = batch_base + new_row + ti;
      const uint32_t another = batch_base + new_col + tj;
      const int out_idx = (new_row + ti) * kStride + kColOff + new_col + tj;
      float ip = ip_tile[warp_id][e];
      state->news_dist[out_idx] = -2 * ip + state->vector_powers[one] + state->vector_powers[another];
      state->news_rank[out_idx] = another;
    }
  }

  if (new_count % 16 != 0) {
    const int wmma_rows = (new_count / 16) * 16;
    const int wmma_cols = (new_count / 16) * 16;
    const int total = new_count * new_count;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
      const int i = idx / new_count;
      const int j = idx % new_count;
      if (i < wmma_rows && j < wmma_cols) continue;
      float ip = 0.0f;
      const int row_st = i * state->vector_dim;
      const int col_st = j * state->vector_dim;
      for (int k = 0; k < state->vector_dim; ++k) {
        ip += __half2float(new_vector_store[row_st + k]) * __half2float(new_vector_store[col_st + k]);
      }
      const int out_idx = i * kStride + kColOff + j;
      const uint32_t one = batch_base + i;
      const uint32_t another = batch_base + j;
      state->news_dist[out_idx] = -2 * ip + state->vector_powers[one] + state->vector_powers[another];
      state->news_rank[out_idx] = another;
    }
  }
}

#ifndef NDEBUG
constexpr int kPrunedMaskCap = LEVEL_SZ_THRES + BATCHSZ_PER_NEW;

__device__ inline void debug_check_pruned_mask_range(uint32_t n, const char *where, int vid, int lv) {
  if (n > static_cast<uint32_t>(kPrunedMaskCap)) {
    printf(
        "Fatal: pruned_mask OOB at %s n=%u cap=%d vid=%d lv=%d block=%u tid=%u\n",
        where,
        n,
        kPrunedMaskCap,
        vid,
        lv,
        blockIdx.x,
        threadIdx.x);
    assert(false);
  }
}
#endif

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
#ifndef NDEBUG
      debug_check_pruned_mask_range(state->old_vec_fetch_offset, "connect_new_to_old", vid, lv);
#endif
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
        const float *prev_vec = state->vector_data + static_cast<size_t>(prev_neigh_id) * state->vector_dim;
        for (int i = ty; i < state->old_vec_fetch_offset; i += nrow) {
          if (i <= prev_neigh_rank) continue;
          if (pruned_mask[i] == 0) {
            can_continue = 1;
            float dist = 0.0f;
            int cand = ranked_cand[i];
            const float *cand_vec = state->vector_data + static_cast<size_t>(cand) * state->vector_dim;
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
        datal[i] = cand;
        distl[i] = ranked_dist[neigh_rank[i]];
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        setListCount(linkl, curr_neigh_cnt);
      }
      __syncthreads();
    }
  }
}

// Merge in_ids/in_dists with the new-new panel in news_*[LEVEL_SZ_THRES..] and
// publish <= M neighbors into the new vector's link list at lv.
// ranked_cand/ranked_dist are the news_* prefix for this new vector.
// Callers own the shared scratch and must be one block per new vector.
__device__ void combine_prune_one_level(
    GpuGraphState *state,
    int vid,
    int lv,
    uint32_t *ranked_cand,
    float *ranked_dist,
    const uint32_t *in_ids,
    const float *in_dists,
    uint32_t in_sz,
    uint32_t *shared_sz,
    uint32_t *prev_neigh_rank,
    uint32_t *prev_neigh_id,
    volatile uint32_t *curr_neigh_cnt,
    uint32_t *neigh_rank,
    unsigned char *pruned_mask,
    bool *can_continue) {
  const int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  uint32_t *linkl = get_level_linklist(state, vid, lv);
  uint32_t *datal = (uint32_t *)(linkl + 1);
  float *distl = get_level_linklist_dist(state, vid, lv);
  if (threadIdx.x == 0) {
    *shared_sz = in_sz;
  }
  for (int i = threadIdx.x; i < static_cast<int>(in_sz); i += blockDim.x) {
    ranked_dist[i] = in_dists[i];
    ranked_cand[i] = in_ids[i];
  }
  __syncthreads();
  static_assert(LEVEL_SZ_THRES - MAX_M0 >= BATCHSZ_PER_NEW);  // Avoid overwriting when moving data.
  for (int i = threadIdx.x; i < new_count; i += blockDim.x) {
    if (state->element_levels[state->cur_element_count + i] >= lv) {
      uint32_t pos = atomicAdd(shared_sz, 1);
      ranked_dist[pos] = ranked_dist[i + LEVEL_SZ_THRES];
      ranked_cand[pos] = ranked_cand[i + LEVEL_SZ_THRES];
    }
  }
  __syncthreads();
  uint32_t sz = *shared_sz;

  int M = lv ? state->M : state->maxM0;
  // No heuristic prune needed. Still drop the self-edge and publish count
  // only after neighbor ids are written. Do not assume ranked_cand[0] is
  // self: this path has not sorted, and ranked_cand[sz] is unfilled.
  if (sz < static_cast<uint32_t>(M) + 1) {
    if (threadIdx.x == 0) {
      uint32_t out = 0;
      for (uint32_t i = 0; i < sz; ++i) {
        const uint32_t cand = ranked_cand[i];
        if (cand == static_cast<uint32_t>(vid)) {
          continue;
        }
        datal[out] = cand;
        distl[out] = ranked_dist[i];
        ++out;
      }
      setListCount(linkl, out);
    }
    __syncthreads();
    return;
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

  // Heuristic prune only the closest ef_construction (CPU HNSW does the
  // same on the search hit list). The bitonic pass already ranked the
  // full combined list, so the prefix is the nearest efc.
  if (sz > state->ef_construction) {
    sz = state->ef_construction;
  }

#ifndef NDEBUG
  debug_check_pruned_mask_range(sz, "combine_prune_one_level", vid, lv);
#endif
  for (int i = threadIdx.x; i < sz; i += blockDim.x) {
    pruned_mask[i] = (ranked_cand[i] == vid) ? 1 : 0;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    *curr_neigh_cnt = 0;
    for (int i = 0; i < sz; i++) {
      if (pruned_mask[i] == 0) {
        *prev_neigh_rank = i;
        *prev_neigh_id = ranked_cand[i];
        neigh_rank[0] = *prev_neigh_rank;
        *curr_neigh_cnt = 1;
        pruned_mask[i] = 1;
        break;
      }
    }
  }
  __syncthreads();
  if (*curr_neigh_cnt == 0) {
    if (threadIdx.x == 0) {
      setListCount(linkl, 0);
    }
    __syncthreads();
    return;
  }

  while (*curr_neigh_cnt < M) {
    // Start to treat the block as 2d, x for dimensions, y for candidates
    int tx = threadIdx.x % warpSize;  // compute distance along this dimension
    int ty = threadIdx.x / warpSize;  // compute for different candidates
    int nrow = blockDim.x / warpSize;

    if (threadIdx.x == 0) {
      *can_continue = 0;
    }
    __syncthreads();
    const float *prev_vec = state->vector_data + static_cast<size_t>(*prev_neigh_id) * state->vector_dim;
    for (int i = ty; i < sz; i += nrow) {
      if (i <= *prev_neigh_rank) continue;
      if (pruned_mask[i] == 0) {
        *can_continue = 1;
        float dist = 0.0f;
        int cand = ranked_cand[i];
        const float *cand_vec = state->vector_data + static_cast<size_t>(cand) * state->vector_dim;
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
    if (!*can_continue) {
      break;
    }
    // Re-treat as 1d, adding the first survived edge.
    if (threadIdx.x == 0) {
      for (int i = *prev_neigh_rank + 1; i < sz; i++) {
        if (pruned_mask[i] == 0) {
          *prev_neigh_rank = i;
          *prev_neigh_id = ranked_cand[i];
          neigh_rank[*curr_neigh_cnt] = i;
          (*curr_neigh_cnt)++;
          pruned_mask[i] = 1;
          break;
        }
      }
    }
    __syncthreads();
  }

  // Still treat as 1d block, adding all the selected edges based on the result.
  if (threadIdx.x == 0) {
    uint32_t out = 0;
    for (uint32_t i = 0; i < *curr_neigh_cnt; ++i) {
      const uint32_t cand = ranked_cand[neigh_rank[i]];
      if (cand == static_cast<uint32_t>(vid) || cand >= state->max_elements ||
          state->element_levels[cand] < static_cast<uint32_t>(lv)) {
        continue;
      }
      datal[out] = cand;
      distl[out] = ranked_dist[neigh_rank[i]];
      ++out;
    }
    setListCount(linkl, out);
  }
  __syncthreads();
}

// 1 block for 1 new vector. Lower levels are already pruned in search.
__device__ void combine_prune_for_new_kernel(GpuGraphState *state, int startup_level) {
  __shared__ uint32_t prev_neigh_rank;
  __shared__ uint32_t prev_neigh_id;
  __shared__ volatile uint32_t curr_neigh_cnt;
  __shared__ uint32_t neigh_rank[MAX_M0];
  __shared__ unsigned char pruned_mask[TOPQ_SZ + BATCHSZ_PER_NEW];
  __shared__ bool can_continue;
  __shared__ uint32_t shared_sz;
  int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  for (int bid = blockIdx.x; bid < new_count; bid += gridDim.x) {
    const int vid = state->cur_element_count + bid;
    auto ranked_cand = state->news_rank + (bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW));
    auto ranked_dist = state->news_dist + (bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW));

    for (int lv = startup_level; lv <= state->element_levels[vid]; ++lv) {
      uint32_t *linkl = get_level_linklist(state, vid, lv);
      const uint32_t sz = getListCount(linkl);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = get_level_linklist_dist(state, vid, lv);
      combine_prune_one_level(
          state,
          vid,
          lv,
          ranked_cand,
          ranked_dist,
          datal,
          distl,
          sz,
          &shared_sz,
          &prev_neigh_rank,
          &prev_neigh_id,
          &curr_neigh_cnt,
          neigh_rank,
          pruned_mask,
          &can_continue);
    }
  }
}

// 1 block per new vector; reverse edges for every level after combine-prune.
__device__ void add_reverse_edges_kernel(GpuGraphState *state) {
  const int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  for (int bid = blockIdx.x; bid < new_count; bid += gridDim.x) {
    const int vid = state->cur_element_count + bid;
#ifndef NDEBUG
    if (vid < 0 || static_cast<uint32_t>(vid) >= state->max_elements) {
      printf(
          "Fatal: add_reverse: vid %d out of range (max_elements=%u cur=%u bid=%d)\n",
          vid,
          state->max_elements,
          state->cur_element_count,
          bid);
      assert(false);
      continue;
    }
#endif
    const int max_lv = static_cast<int>(state->element_levels[vid]);
    for (int lv = 0; lv <= max_lv; ++lv) {
#ifndef NDEBUG
      const uint32_t capacity = link_capacity_at_level(state, lv);
      if (lv > 0 && (state->link_lists == nullptr || state->link_lists[vid] == nullptr)) {
        printf(
            "Fatal: add_reverse: new %d null link_lists at level %d (element_level=%u)\n",
            vid,
            lv,
            state->element_levels[vid]);
        assert(false);
        continue;
      }
#endif
      uint32_t *linkl = get_level_linklist(state, vid, lv);
#ifndef NDEBUG
      if (linkl == nullptr) {
        printf("Fatal: add_reverse: new %d null linkl at level %d\n", vid, lv);
        assert(false);
        continue;
      }
#endif
      const uint32_t sz = getListCount(linkl);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = get_level_linklist_dist(state, vid, lv);
#ifndef NDEBUG
      if (sz > capacity) {
        printf("Fatal: add_reverse: new %d level %d count %u exceeds capacity %u\n", vid, lv, sz, capacity);
        assert(false);
        continue;
      }
#endif
      for (uint32_t i = threadIdx.x; i < sz; i += blockDim.x) {
        const uint32_t other = datal[i];
#ifndef NDEBUG
        if (other >= state->max_elements) {
          printf(
              "Fatal: add_reverse: new %d level %d neighbor[%u]=%u >= max_elements %u\n",
              vid,
              lv,
              i,
              other,
              state->max_elements);
          assert(false);
          continue;
        }
        if (other >= state->cur_element_count + static_cast<uint32_t>(new_count)) {
          printf(
              "Fatal: add_reverse: new %d level %d neighbor[%u]=%u outside current batch "
              "(cur=%u new_count=%d)\n",
              vid,
              lv,
              i,
              other,
              state->cur_element_count,
              new_count);
          assert(false);
          continue;
        }
#endif
        if (other >= state->cur_element_count) {
          continue;
        }
#ifndef NDEBUG
        if (state->element_levels[other] < static_cast<uint32_t>(lv)) {
          printf(
              "Fatal: add_reverse: other %u not on level %d (element_level=%u) from new %d idx %u\n",
              other,
              lv,
              state->element_levels[other],
              vid,
              i);
          assert(false);
          continue;
        }
        if (lv > 0 && (state->link_lists == nullptr || state->link_lists[other] == nullptr)) {
          printf("Fatal: add_reverse: other %u null link_lists at level %d from new %d\n", other, lv, vid);
          assert(false);
          continue;
        }
#endif
        uint32_t *other_linkl = get_level_linklist(state, other, lv);
        uint32_t *other_datal = (uint32_t *)(other_linkl) + 1;
        float *other_distl = get_level_linklist_dist(state, other, lv);
#ifndef NDEBUG
        if (other_linkl == nullptr) {
          printf("Fatal: add_reverse: other %u null other_linkl at level %d from new %d\n", other, lv, vid);
          assert(false);
          continue;
        }
#endif
        const uint32_t pos = atomicAdd((uint32_t *)(other_linkl), 1);
        record_changed_old_link(state, other, lv);
#ifndef NDEBUG
        if (pos >= capacity) {
          printf(
              "Fatal: reverse link list out of bound at level %d for node %u "
              "(pos=%u capacity=%u new=%d sz=%u idx=%u)\n",
              lv,
              other,
              pos,
              capacity,
              vid,
              sz,
              i);
          assert(false);
          continue;
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

__global__ void build_compute_level_delimiter_kernel(GpuGraphState *state) {
  for (int level = 1; level < MAX_HNSW_LEVEL; ++level) {
    state->old_vector_level_delimiter[level] += state->old_vector_level_delimiter[level - 1];
  }
}

__global__ void build_dist_old_new_and_new_new_kernel(GpuGraphState *state) {
  uint64_t phase_t0 = build_phase_begin(state);
  // Compute the distances between the new vectors and the old vectors.
  compute_dist_with_old_kernel(state);
  build_phase_record(state, kBuildPhaseDistOldNew, phase_t0);

  phase_t0 = build_phase_begin(state);
  // Compute the distances among the new vectors for this batch into news_dist.
  compute_new_new_dist_kernel(state);
  build_phase_record(state, kBuildPhaseDistNewNew, phase_t0);
}

__global__ void build_sort_and_connect_upper_kernel(GpuGraphState *state) {
  const int startup_level = compute_startup_level(state);
  uint64_t phase_t0 = build_phase_begin(state);
  // Every new vector needs the closest old vector as the lower-search entry.
  // Only vectors that reach startup_level need the complete ordering for upper pruning.
  bitonic_sort_id_by_dis(state, startup_level);
  build_phase_record(state, kBuildPhaseSortOldByDist, phase_t0);

  phase_t0 = build_phase_begin(state);
  // Connect the new vectors to the old vectors at upper levels (>= startup_level).
  connect_new_to_old_at_upper_kernel(state, startup_level);
  build_phase_record(state, kBuildPhaseConnectUpper, phase_t0);
}

__global__ void build_snapshot_frozen_kernel(GpuGraphState *state) {
  const int startup_level = compute_startup_level(state);
  uint64_t phase_t0 = build_phase_begin(state);
  // Freeze the link counts used while searching lower levels.
  snapshot_frozen_link_counts_kernel(state, startup_level);
  build_phase_record(state, kBuildPhaseSnapshotFrozen, phase_t0);
}

__global__ void build_search_knn_lower_kernel(GpuGraphState *state) {
  const int startup_level = compute_startup_level(state);
  uint64_t phase_t0 = build_phase_begin(state);
  search_knn_at_lower_kernel(state, startup_level);
  build_phase_record(state, kBuildPhaseSearchPruneLower, phase_t0);
}

__global__ void build_combine_prune_new_kernel(GpuGraphState *state) {
  const int startup_level = compute_startup_level(state);
  uint64_t phase_t0 = build_phase_begin(state);
  combine_prune_for_new_kernel(state, startup_level);
  build_phase_record(state, kBuildPhaseCombinePruneUpper, phase_t0);
}

__global__ void build_add_reverse_kernel(GpuGraphState *state) {
  uint64_t phase_t0 = build_phase_begin(state);
  add_reverse_edges_kernel(state);
  build_phase_record(state, kBuildPhaseAddReverse, phase_t0);
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
  auto sync_kernel = [](const char *name, uint32_t batch) -> cudaError_t {
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("CUDA error after %s (batch start %u): %s\n", name, batch, cudaGetErrorString(err));
    }
    return err;
  };
  for (uint32_t i = 0; i < max_elements; i += BATCHSZ_PER_NEW) {
    build_reset_batch_state_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    if (const cudaError_t err = sync_kernel("build_reset_batch_state_kernel", i); err != cudaSuccess) return err;
    build_aggregate_on_level_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    if (const cudaError_t err = sync_kernel("build_aggregate_on_level_kernel", i); err != cudaSuccess) return err;
    build_compute_level_delimiter_kernel<<<1, 1>>>(state);
    build_dist_old_new_and_new_new_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    if (const cudaError_t err = sync_kernel("build_dist_old_new_and_new_new_kernel", i); err != cudaSuccess) return err;
    build_sort_and_connect_upper_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    if (const cudaError_t err = sync_kernel("build_sort_and_connect_upper_kernel", i); err != cudaSuccess) return err;
    // Reverse edges are only appended in build_add_reverse_kernel, after search.
    // Old lists are stable during search; get_frozen_link_count reads live size.
    // build_snapshot_frozen_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    // if (const cudaError_t err = sync_kernel("build_snapshot_frozen_kernel", i); err != cudaSuccess) return err;
    // news_*[0] needs to be used as the entry; then news_* become the scratch pad.
    build_search_knn_lower_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    if (const cudaError_t err = sync_kernel("build_search_knn_lower_kernel", i); err != cudaSuccess) return err;
    build_combine_prune_new_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    if (const cudaError_t err = sync_kernel("build_combine_prune_new_kernel", i); err != cudaSuccess) return err;
    build_add_reverse_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    if (const cudaError_t err = sync_kernel("build_add_reverse_kernel", i); err != cudaSuccess) return err;
    build_sort_prune_old_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    if (const cudaError_t err = sync_kernel("build_sort_prune_old_kernel", i); err != cudaSuccess) return err;
    build_update_batch_kernel<<<GRID_DIM, BLOCK_DIM>>>(state);
    if (const cudaError_t err = sync_kernel("build_update_batch_kernel", i); err != cudaSuccess) return err;
  }
  return cudaSuccess;
}

// 1d grid, 1d block
__device__ void compute_power_kernel(GpuGraphState *state) {
  __shared__ float cache[MAX_DIM];
  int bid = blockIdx.x;
  while (bid < state->max_elements) {
    size_t vector_start = static_cast<size_t>(bid) * state->vector_dim;
    size_t vector_end = vector_start + state->vector_dim;
    size_t wid = vector_start + threadIdx.x;
    int cid = threadIdx.x;
    float temp = 0.0f;
    // TODO: Rewrite this to intra-warp reduction.
    while (wid < vector_end) {
      temp += state->vector_data[wid] * state->vector_data[wid];
      wid += blockDim.x;
    }
    cache[cid] = temp;
    __syncthreads();

    int active_threads = blockDim.x;
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
  half *new_vector_store = state->half_vector_data + static_cast<size_t>(state->cur_element_count) * state->vector_dim;
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
      const size_t vector_st = static_cast<size_t>(state->old_vector_fetch_index[i + oid]) * state->vector_dim;
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
#ifndef NDEBUG
        debug_check_pruned_mask_range(static_cast<uint32_t>(sz), "prune_for_old", static_cast<int>(vid), lv);
#endif
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
          const float *prev_vec = state->vector_data + static_cast<size_t>(prev_neigh_id) * state->vector_dim;

          for (int i = ty; i < sz; i += nrow) {
            if (i <= prev_neigh_rank) continue;
            if (pruned_mask[i] == 0) {
              can_continue = 1;
              float dist = 0.0f;
              const int cand = datal[i];
              const float *cand_vec = state->vector_data + static_cast<size_t>(cand) * state->vector_dim;
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

#ifndef NDEBUG
// Illegal-access guards for search_knn_at_lower_kernel. Check node identity and
// upper-layer pointer *before* get_level_linklist / vector_data / visited loads.
__device__ void debug_search_lower_check_id(GpuGraphState *state, uint32_t id, int vid, int lv, const char *where) {
  if (id >= state->max_elements) {
    printf(
        "Fatal: search_knn_at_lower: %s id %u >= max_elements %u "
        "(vid=%d lv=%d cur=%u block=%u tid=%u)\n",
        where,
        id,
        state->max_elements,
        vid,
        lv,
        state->cur_element_count,
        blockIdx.x,
        threadIdx.x);
    assert(false);
  }
}

__device__ void
debug_search_lower_check_node_level(GpuGraphState *state, uint32_t id, int lv, int vid, const char *where) {
  debug_search_lower_check_id(state, id, vid, lv, where);
  if (lv < 0 || lv >= MAX_HNSW_LEVEL) {
    printf("Fatal: search_knn_at_lower: %s level %d out of range for node %u (vid=%d)\n", where, lv, id, vid);
    assert(false);
  }
  if (state->element_levels[id] < static_cast<uint32_t>(lv)) {
    printf(
        "Fatal: search_knn_at_lower: %s node %u not on level %d (element_level=%u vid=%d tid=%u)\n",
        where,
        id,
        lv,
        state->element_levels[id],
        vid,
        threadIdx.x);
    assert(false);
  }
  if (lv > 0 && (state->link_lists == nullptr || state->link_lists[id] == nullptr)) {
    printf(
        "Fatal: search_knn_at_lower: %s null link_lists for node %u at level %d "
        "(element_level=%u vid=%d)\n",
        where,
        id,
        lv,
        state->element_levels[id],
        vid);
    assert(false);
  }
}

__device__ void debug_search_lower_check_list(
    GpuGraphState *state,
    uint32_t id,
    int lv,
    uint32_t size,
    uint32_t *linkl,
    int vid,
    const char *where) {
  if (linkl == nullptr) {
    printf("Fatal: search_knn_at_lower: %s null linkl for node %u at level %d (vid=%d)\n", where, id, lv, vid);
    assert(false);
  }
  const uint32_t capacity = link_capacity_at_level(state, lv);
  if (size > capacity) {
    printf(
        "Fatal: search_knn_at_lower: %s node %u level %d size %u exceeds capacity %u (vid=%d)\n",
        where,
        id,
        lv,
        size,
        capacity,
        vid);
    assert(false);
  }
}

__device__ void
debug_search_lower_check_cand(GpuGraphState *state, uint32_t cand, uint32_t from, int lv, int vid, const char *where) {
  if (cand >= state->max_elements) {
    printf(
        "Fatal: search_knn_at_lower: %s cand %u >= max_elements %u from node %u "
        "at lv %d (vid=%d)\n",
        where,
        cand,
        state->max_elements,
        from,
        lv,
        vid);
    assert(false);
  }
  if (state->element_levels[cand] < static_cast<uint32_t>(lv)) {
    printf(
        "Fatal: search_knn_at_lower: %s off-level cand %u (element_level=%u) at lv %d "
        "from node %u (vid=%d)\n",
        where,
        cand,
        state->element_levels[cand],
        lv,
        from,
        vid);
    assert(false);
  }
}
#endif  // NDEBUG

// Worker blocks claim new vectors from search_next_bid.
// Threads in a block compute distances for the claimed vector.
__device__ void search_knn_at_lower_kernel(GpuGraphState *state, const int startup_lv) {
  __shared__ uint32_t *visited;
  __shared__ uint32_t visited_tag;
  __shared__ int visited_wrap;
  __shared__ Neighbor candq[CANDQ_SZ];
  __shared__ volatile int candq_sz;
  __shared__ Neighbor topq[TOPQ_SZ];
  __shared__ volatile int topq_sz;
  __shared__ float topq_max;
  __shared__ uint32_t curr_obj_shared;
  __shared__ int curr_dist_bits_shared;
  __shared__ Neighbor warp_staging[SEARCH_WARP_COUNT][WARP_STAGING_CAP];
  __shared__ int warp_staging_sz[SEARCH_WARP_COUNT];
  __shared__ float warp_best_dist[SEARCH_WARP_COUNT];
  __shared__ uint32_t warp_best_cand[SEARCH_WARP_COUNT];
  __shared__ Neighbor popped;
  __shared__ int claimed;
  __shared__ uint32_t prune_prev_neigh_rank;
  __shared__ uint32_t prune_prev_neigh_id;
  __shared__ volatile uint32_t prune_curr_neigh_cnt;
  __shared__ uint32_t prune_neigh_rank[MAX_M0];
  __shared__ unsigned char prune_mask[TOPQ_SZ + BATCHSZ_PER_NEW];
  __shared__ bool prune_can_continue;
  __shared__ uint32_t prune_shared_sz;
  const int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
#ifndef NDEBUG
  if (threadIdx.x == 0) {
    if (state->visited == nullptr || state->visited_tags == nullptr) {
      printf("Fatal: search_knn_at_lower: visited/visited_tags is null\n");
      assert(false);
    }
    if (state->vector_data == nullptr) {
      printf("Fatal: search_knn_at_lower: vector_data is null\n");
      assert(false);
    }
    if (state->news_rank == nullptr || state->news_dist == nullptr) {
      printf("Fatal: search_knn_at_lower: news_rank/news_dist is null\n");
      assert(false);
    }
    if (blockIdx.x >= static_cast<uint32_t>(GRID_DIM)) {
      printf("Fatal: search_knn_at_lower: blockIdx %u >= GRID_DIM %d\n", blockIdx.x, GRID_DIM);
      assert(false);
    }
  }
  __syncthreads();
#endif
  if (threadIdx.x == 0) {
    visited = state->visited + blockIdx.x * state->max_elements;
    visited_tag = state->visited_tags[blockIdx.x];
    visited_wrap = 0;
  }
  __syncthreads();
#ifdef PROFILE_BUILD_PHASES
  uint64_t n_expand = 0;
  uint64_t clear_cycles = 0;
  uint64_t t_mark = 0;
  if (state->profile_build_phases && threadIdx.x == 0) t_mark = clock64();
#endif
  int steal_phase = 0;
  while (true) {
    if (threadIdx.x == 0) {
      if (steal_phase == 0) {
        claimed = blockIdx.x;
      } else {
        claimed = GRID_DIM + atomicAdd(&state->next_bid, 1U);
      }
    }
    __syncthreads();
    const int bid = claimed;
    if (bid >= new_count) {
      break;
    }
    steal_phase = 1;
    const int vid = state->cur_element_count + bid;
#ifndef NDEBUG
    if (vid < 0 || static_cast<uint32_t>(vid) >= state->max_elements) {
      printf(
          "Fatal: search_knn_at_lower: vid %d out of range (max_elements=%u cur=%u bid=%d)\n",
          vid,
          state->max_elements,
          state->cur_element_count,
          bid);
      assert(false);
    }
#endif
    const int lvl = state->element_levels[vid];
    const int tx = threadIdx.x % warpSize;
    const int ty = threadIdx.x / warpSize;
    const int nrow = blockDim.x / warpSize;

    float *query_vec = state->vector_data + static_cast<size_t>(vid) * state->vector_dim;
    // Find the entry point
    if (threadIdx.x == 0) {
      auto ranks = state->news_rank + bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
      auto dists = state->news_dist + bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
      curr_obj_shared = ranks[0];
      curr_dist_bits_shared = __float_as_int(dists[0]);
#ifndef NDEBUG
      if (curr_obj_shared >= state->max_elements) {
        printf(
            "Fatal: search_knn_at_lower: entry ranks[0]=%u >= max_elements %u "
            "(vid=%d bid=%d old_vec_fetch_offset=%u)\n",
            curr_obj_shared,
            state->max_elements,
            vid,
            bid,
            state->old_vec_fetch_offset);
        assert(false);
      }
      {
        const int entry_lv = startup_lv - 1;
        if (entry_lv > 0 && state->element_levels[curr_obj_shared] < static_cast<uint32_t>(entry_lv)) {
          printf(
              "Fatal: search_knn_at_lower: entry ranks[0]=%u not on level %d "
              "(element_level=%u vid=%d)\n",
              curr_obj_shared,
              entry_lv,
              state->element_levels[curr_obj_shared],
              vid);
          assert(false);
        }
      }
#endif
    }
    __syncthreads();
    // Per-thread copy of the current object. Avoid expanding through a shared
    // `popped` / `curr_obj_shared` that other threads can keep from a previous
    // bid (lv=0 node or uninitialized 0x01010101) after the handshake.
    uint32_t curr_obj = curr_obj_shared;
    float curr_dist = __int_as_float(curr_dist_bits_shared);

    int lv = startup_lv - 1;
    while (lv > lvl) {
      while (1) {
#ifndef NDEBUG
        if (threadIdx.x == 0 && state->old_vec_fetch_offset == 0) {
          printf(
              "Fatal: search_knn_at_lower: old_vec_fetch_offset == 0\n"
              "Startup level %d, vid %d, element level %d\n",
              startup_lv,
              vid,
              lvl);
          assert(false);
        }
        debug_search_lower_check_node_level(state, curr_obj, lv, vid, "greedy expand");
#endif
        const int size = get_frozen_link_count(state, curr_obj_shared, lv);
        uint32_t *linkl = get_level_linklist(state, curr_obj, lv);
        uint32_t *datal = (uint32_t *)(linkl + 1);
#ifndef NDEBUG
        debug_search_lower_check_list(state, curr_obj, lv, static_cast<uint32_t>(size), linkl, vid, "greedy expand");
#endif

        float local_best;
        uint32_t local_cand;
        if (tx == 0) {
          local_best = curr_dist;
          local_cand = curr_obj;
        }
        for (int i = ty; i < size; i += nrow) {
          uint32_t cand = datal[i];
#ifndef NDEBUG
          debug_search_lower_check_cand(state, cand, curr_obj, lv, vid, "greedy neighbor");
#endif
          float dist = 0.0f;
          for (int j = tx; j < state->vector_dim; j += warpSize) {
            const float diff = state->vector_data[static_cast<size_t>(cand) * state->vector_dim + j] - query_vec[j];
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
        float best_d = curr_dist;
        uint32_t best_c = curr_obj;
        int greedy_changed = 0;
        for (int w = 0; w < SEARCH_WARP_COUNT; ++w) {
          if (warp_best_dist[w] < best_d) {
            best_d = warp_best_dist[w];
            best_c = warp_best_cand[w];
            greedy_changed = 1;
          }
        }
        curr_dist = best_d;
        curr_obj = best_c;
        if (threadIdx.x == 0) {
          curr_obj_shared = curr_obj;
          curr_dist_bits_shared = __float_as_int(curr_dist);
        }
        __syncthreads();
        if (!greedy_changed) {
          break;
        }
      }
      lv--;
    }
    // Search and insert at current level.
    if (lv > lvl) {  // In case the greedy expansion ends due to break.
      lv = lvl;
    }
    while (lv >= 0) {
      if (threadIdx.x == 0) {
#ifndef NDEBUG
        debug_search_lower_check_id(state, curr_obj, vid, lv, "beam entry");
#endif
        candq_sz = 0;
        MinPqPush(candq, &candq_sz, curr_dist, curr_obj, 0);
        topq_sz = 0;
        MaxPqPush(topq, &topq_sz, curr_dist, curr_obj, 1);
        topq_max = curr_dist;
        ++visited_tag;
        visited_wrap = (visited_tag == 0);
        if (visited_wrap) visited_tag = 1;
      }
      __syncthreads();
      if (visited_wrap) {
#ifdef PROFILE_BUILD_PHASES
        uint64_t t_clear = 0;
        if (state->profile_build_phases && threadIdx.x == 0) t_clear = clock64();
#endif
        for (int i = threadIdx.x; i < state->max_elements; i += blockDim.x) {
          visited[i] = 0;
        }
        __syncthreads();
#ifdef PROFILE_BUILD_PHASES
        if (state->profile_build_phases && threadIdx.x == 0) {
          clear_cycles += clock64() - t_clear;
        }
#endif
      }

      while (true) {
        // All threads compute the stop test from already-volatile queue sizes
        // so they cannot diverge on a stale shared `search_continue`.
        const int csz = candq_sz;
        const int tsz = topq_sz;
        const int efc = static_cast<int>(state->ef_construction);
        const int search_cont = (csz > 0) && !(tsz >= efc && candq[0].distance > topq[0].distance);
        if (!search_cont) {
          break;
        }
#ifdef PROFILE_BUILD_PHASES
        if (state->profile_build_phases && threadIdx.x == 0) {
          ++n_expand;
        }
#endif

        // Read the heap root *before* thread 0 pops. nodeid is already volatile.
        const uint32_t node = static_cast<uint32_t>(candq[0].nodeid);
#ifndef NDEBUG
        debug_search_lower_check_node_level(state, node, lv, vid, "beam expand");
#endif
        if (threadIdx.x == 0) {
          MinPqPop(candq, &candq_sz, &popped);
          visited[node] = visited_tag;
        }
        __syncthreads();

        const int tx = threadIdx.x % warpSize;
        const int ty = threadIdx.x / warpSize;
        const int nrow = blockDim.x / warpSize;

        if (tx == 0) {
          warp_staging_sz[ty] = 0;
        }
        const int size = get_frozen_link_count(state, node, lv);
        uint32_t *linkl = get_level_linklist(state, node, lv);
        uint32_t *datal = (uint32_t *)(linkl + 1);
#ifndef NDEBUG
        debug_search_lower_check_list(state, node, lv, size, linkl, vid, "beam expand");
#endif
        for (int i = ty; i < size; i += nrow) {
          uint32_t cand = datal[i];
#ifndef NDEBUG
          debug_search_lower_check_cand(state, cand, node, lv, vid, "beam neighbor");
#endif
          // uint32_t prev_tag = 0;
          // if (tx == 0) {
          //   prev_tag = atomicExch(&visited[cand], visited_tag);
          // }
          // prev_tag = __shfl_sync(0xffffffff, prev_tag, 0);
          uint32_t prev_tag = visited[cand];
          if (prev_tag == visited_tag) {
            continue;
          }
          visited[cand] = visited_tag;
          float dist = 0.0f;
          for (int j = tx; j < state->vector_dim; j += warpSize) {
            const float diff = state->vector_data[static_cast<size_t>(cand) * state->vector_dim + j] - query_vec[j];
            dist += diff * diff;
          }
          for (int lane = warpSize / 2; lane > 0; lane /= 2) {
            dist += __shfl_down_sync(0xffffffff, dist, lane);
          }
          if (tx == 0) {
            const int pos = warp_staging_sz[ty];
            if (pos < WARP_STAGING_CAP) {
#ifndef NDEBUG
              if (cand >= state->max_elements) {
                printf(
                    "Fatal: search_knn_at_lower: cand %u out of range "
                    "(vid=%d lv=%d max_elements=%u)\n",
                    cand,
                    vid,
                    lv,
                    state->max_elements);
                assert(false);
              }
#endif
              warp_staging[ty][pos].distance = dist;
              warp_staging[ty][pos].nodeid = cand;
              warp_staging_sz[ty] = pos + 1;
            }
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          merge_staged_neighbors_into_queues(
              candq, &candq_sz, topq, &topq_sz, &topq_max, warp_staging, warp_staging_sz, state->ef_construction);
#ifndef NDEBUG
          if (candq_sz < 0 || candq_sz > CANDQ_SZ || topq_sz < 0 || topq_sz > TOPQ_SZ) {
            printf(
                "Fatal: search_knn_at_lower: queue overflow after merge "
                "(vid=%d lv=%d candq_sz=%d/%d topq_sz=%d/%d)\n",
                vid,
                lv,
                candq_sz,
                CANDQ_SZ,
                topq_sz,
                TOPQ_SZ);
            assert(false);
          }
#endif
        }
        __syncthreads();
      }

#ifndef NDEBUG
      debug_search_lower_check_node_level(state, static_cast<uint32_t>(vid), lv, vid, "writeback new");
#endif
      // Next-level entry is the closest node in the result set (topq), not the
      // leftover candidate heap (which can be empty after a full expand).
      if (topq_sz > 0) {
        uint32_t entry_id = curr_obj;
        float entry_dist = curr_dist;
        find_closest_in_queue(topq, topq_sz, &entry_id, &entry_dist);
        curr_obj = entry_id;
        curr_dist = entry_dist;
      }
      auto ranked_cand = state->news_rank + bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
      auto ranked_dist = state->news_dist + bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
      const int write_sz = min(topq_sz, TOPQ_SZ);
      for (int i = threadIdx.x; i < write_sz; i += blockDim.x) {
        ranked_cand[i] = topq[i].nodeid;
        ranked_dist[i] = topq[i].distance;
      }
      __syncthreads();
      combine_prune_one_level(
          state,
          vid,
          lv,
          ranked_cand,
          ranked_dist,
          ranked_cand,
          ranked_dist,
          static_cast<uint32_t>(write_sz),
          &prune_shared_sz,
          &prune_prev_neigh_rank,
          &prune_prev_neigh_id,
          &prune_curr_neigh_cnt,
          prune_neigh_rank,
          prune_mask,
          &prune_can_continue);
#ifndef NDEBUG
      {
        uint32_t *linkl = get_level_linklist(state, vid, lv);
        const uint32_t published = getListCount(linkl);
        const uint32_t M = lv ? state->M : state->maxM0;
        if (published > M) {
          printf(
              "Fatal: search_knn_at_lower: published count %u > M %u "
              "(vid=%d lv=%d topq_sz=%d)\n",
              published,
              M,
              vid,
              lv,
              topq_sz);
          assert(false);
        }
      }
#endif
      if (threadIdx.x == 0) {
        curr_obj_shared = curr_obj;
        curr_dist_bits_shared = __float_as_int(curr_dist);
#ifndef NDEBUG
        if (curr_obj >= state->max_elements) {
          printf(
              "Fatal: search_knn_at_lower: next-level entry %u >= max_elements %u "
              "(vid=%d lv=%d topq_sz=%d)\n",
              curr_obj,
              state->max_elements,
              vid,
              lv,
              topq_sz);
          assert(false);
        }
#endif
      }
      lv--;
      __syncthreads();
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    state->visited_tags[blockIdx.x] = visited_tag;
#ifdef PROFILE_BUILD_PHASES
    if (state->profile_build_phases) {
      const uint32_t batch = state->cur_element_count / BATCHSZ_PER_NEW;
      const uint32_t idx = batch * GRID_DIM + blockIdx.x;
      state->search_block_cycles[idx] = clock64() - t_mark;
      state->search_block_clear_cycles[idx] = clear_cycles;
      state->search_block_expands[idx] = n_expand;
    }
#endif
  }
}

#ifdef PROFILE_BUILD_PHASES
#include <fmt/format.h>
#include <algorithm>
#include <vector>

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

namespace {

double percentile_sorted(const std::vector<double> &sorted, double p) {
  if (sorted.empty()) return 0.0;
  const double idx = p * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double frac = idx - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

void summarize_batch_imbalance(const uint64_t *values, uint64_t batch_begin, uint64_t batch_end, const char *label) {
  std::vector<double> ratios;
  ratios.reserve(static_cast<size_t>(batch_end - batch_begin));
  double sum_idle = 0.0;
  double sum_max_mean = 0.0;
  uint64_t global_min = ~0ULL;
  uint64_t global_max = 0;
  for (uint64_t b = batch_begin; b < batch_end; ++b) {
    const uint64_t *row = values + b * GRID_DIM;
    uint64_t mn = ~0ULL;
    uint64_t mx = 0;
    unsigned long long sum = 0;
    for (int blk = 0; blk < GRID_DIM; ++blk) {
      const uint64_t v = row[blk];
      mn = std::min(mn, v);
      mx = std::max(mx, v);
      sum += v;
    }
    const double mean = static_cast<double>(sum) / GRID_DIM;
    const double ratio = mean > 0.0 ? static_cast<double>(mx) / mean : 0.0;
    const double idle = mx > 0 ? 1.0 - mean / static_cast<double>(mx) : 0.0;
    ratios.push_back(ratio);
    sum_idle += idle;
    sum_max_mean += ratio;
    global_min = std::min(global_min, mn);
    global_max = std::max(global_max, mx);
  }
  const double n = static_cast<double>(batch_end - batch_begin);
  std::sort(ratios.begin(), ratios.end());
  fmt::print(
      "  {} batches [{}..{}]: max/mean p50={:.3f} p95={:.3f} avg={:.3f}  "
      "avg idle={:.1f}%  block min/max (any batch)={}/{}\n",
      label,
      batch_begin,
      batch_end - 1,
      percentile_sorted(ratios, 0.50),
      percentile_sorted(ratios, 0.95),
      sum_max_mean / n,
      100.0 * sum_idle / n,
      global_min,
      global_max);
}

}  // namespace

void print_search_block_profile(
    const uint64_t *block_cycles,
    const uint64_t *block_clear_cycles,
    const uint64_t *block_expands,
    uint64_t batch_count) {
  if (batch_count == 0 || block_cycles == nullptr) {
    fmt::print("search_knn load profile: no batches recorded\n");
    return;
  }

  int clock_rate_khz = 0;
  cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
  if (clock_rate_khz <= 0) {
    fmt::print("search_knn load profile: unable to read GPU clock rate\n");
    return;
  }
  const double cycles_per_ms = clock_rate_khz;

  unsigned long long sum_max_search = 0;
  unsigned long long sum_mean_search = 0;
  unsigned long long sum_max_clear = 0;
  unsigned long long sum_expands = 0;
  for (uint64_t b = 0; b < batch_count; ++b) {
    uint64_t mx_search = 0;
    uint64_t mx_clear = 0;
    unsigned long long row_search = 0;
    for (int blk = 0; blk < GRID_DIM; ++blk) {
      const uint64_t search = block_cycles[b * GRID_DIM + blk];
      const uint64_t clear = block_clear_cycles[b * GRID_DIM + blk];
      mx_search = std::max(mx_search, search);
      mx_clear = std::max(mx_clear, clear);
      row_search += search;
      sum_expands += block_expands[b * GRID_DIM + blk];
    }
    sum_max_search += mx_search;
    sum_max_clear += mx_clear;
    sum_mean_search += row_search / GRID_DIM;
  }

  fmt::print(
      "search_knn_at_lower load profile ({} batches, {} blocks, ~{} vectors/block):\n",
      batch_count,
      GRID_DIM,
      BATCHSZ_PER_NEW / GRID_DIM);
  fmt::print(
      "  visited_clear max-block sum {:>10.3f} ms  ({:.3f} ms/batch)  — equal work across blocks\n",
      sum_max_clear / cycles_per_ms,
      (sum_max_clear / cycles_per_ms) / batch_count);
  fmt::print(
      "  search loop   max-block sum {:>10.3f} ms  ({:.3f} ms/batch)  — kernel time follows the slowest block\n",
      sum_max_search / cycles_per_ms,
      (sum_max_search / cycles_per_ms) / batch_count);
  fmt::print(
      "  search loop   mean-block sum {:>10.3f} ms  (mean/max={:.3f}; gap is imbalance tail)\n",
      sum_mean_search / cycles_per_ms,
      sum_max_search > 0 ? static_cast<double>(sum_mean_search) / static_cast<double>(sum_max_search) : 0.0);
  fmt::print(
      "  beam expansions total {}  ({:.1f} per block-batch)\n",
      sum_expands,
      static_cast<double>(sum_expands) / (static_cast<double>(batch_count) * GRID_DIM));

  fmt::print("  cycle imbalance (search loop, per-batch max/mean over {} blocks):\n", GRID_DIM);
  summarize_batch_imbalance(block_cycles, 0, batch_count, "all");
  const uint64_t early_end = std::max<uint64_t>(1, batch_count / 10);
  const uint64_t late_begin = batch_count - early_end;
  summarize_batch_imbalance(block_cycles, 0, early_end, "early 10%");
  summarize_batch_imbalance(block_cycles, late_begin, batch_count, "late 10%");
  fmt::print("  expansion imbalance (per-batch max/mean over {} blocks):\n", GRID_DIM);
  summarize_batch_imbalance(block_expands, 0, batch_count, "all");
  summarize_batch_imbalance(block_expands, 0, early_end, "early 10%");
  summarize_batch_imbalance(block_expands, late_begin, batch_count, "late 10%");
}
#endif  // PROFILE_BUILD_PHASES
