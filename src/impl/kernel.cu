#include <cooperative_groups.h>
#include <stdio.h>
#include <type_traits>
#include "hnswalg-lite.h"
#include "kernel.cuh"

namespace cg = cooperative_groups;

namespace {

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

__device__ uint32_t *get_linklist0(GpuGraphState *state, uint32_t internal_id) {
  return (uint32_t *)(state->level0_links + internal_id * state->size_links_level0);
}

__device__ float *get_linklist_dist0(GpuGraphState *state, uint32_t internal_id) {
  return (float *)(state->level0_links + internal_id * state->size_links_level0 + sizeof(linklistsizeint) +
                   (state->maxM0 + BATCHSZ_PER_NEW) * sizeof(uint32_t));
}

__device__ unsigned short int getListCount(uint32_t *ptr) { return *((uint32_t *)ptr); }

__device__ void setListCount(uint32_t *ptr, unsigned short int size) { *((tableint *)ptr) = size; }

__device__ uint32_t frozen_link_count_offset(GpuGraphState *state, uint32_t internal_id, int level) {
  return static_cast<uint32_t>(level) * state->max_elements + internal_id;
}

__device__ uint32_t get_frozen_link_count(GpuGraphState *state, uint32_t internal_id, int level) {
  if (level < 0 || level >= MAX_HNSW_LEVEL || internal_id >= state->max_elements ||
      state->element_levels[internal_id] < level) {
    return 0;
  }
  return state->frozen_link_counts[frozen_link_count_offset(state, internal_id, level)] & kLinkCountMask;
}

__device__ uint32_t changed_old_link_level_offset(GpuGraphState *state, int level) {
  return static_cast<uint32_t>(level) * static_cast<uint32_t>(BATCHSZ_PER_NEW) * state->maxM0;
}

__device__ uint32_t changed_old_link_level_capacity(GpuGraphState *state) {
  return static_cast<uint32_t>(BATCHSZ_PER_NEW) * state->maxM0;
}

__device__ void record_changed_old_link(GpuGraphState *state, uint32_t internal_id, int level) {
  if (internal_id >= state->cur_element_count || level < 0 || level >= MAX_HNSW_LEVEL) return;

  uint32_t *frozen_count = state->frozen_link_counts + frozen_link_count_offset(state, internal_id, level);
  const uint32_t previous = atomicOr(frozen_count, kChangedOldLinkFlag);
  if ((previous & kChangedOldLinkFlag) != 0) return;

  const uint32_t offset = atomicAdd(&state->changed_old_link_counts[level], 1U);
#ifndef NDEBUG
  if (offset >= changed_old_link_level_capacity(state)) {
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
  uint32_t *linkl = lv == 0 ? get_linklist0(state, vid) : get_linklist(state, vid, lv);
  uint32_t *datal = (uint32_t *)(linkl + 1);
  float *distl = lv == 0 ? (float *)get_linklist_dist0(state, vid) : (float *)get_linklist_dist(state, vid, lv);
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
      uint32_t *linkl = lv == 0 ? get_linklist0(state, vid) : get_linklist(state, vid, lv);
      const int len = getListCount(linkl);
      if (len <= 1 || len == static_cast<int>(get_frozen_link_count(state, vid, lv))) continue;
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

__device__ void aggregate_on_level_kernel(GpuGraphState *state, int lv) {
  int tid = threadIdx.x;
  if (tid == 0) {
    state->old_vec_fetch_offset = 0;
  }
  for (int lv = tid; lv < MAX_HNSW_LEVEL; lv += blockDim.x) {
    state->changed_old_link_counts[lv] = 0;
  }
  __syncthreads();
  for (int i = tid; i < state->cur_element_count; i += blockDim.x) {
    if (state->element_levels[i] >= static_cast<int32_t>(lv)) {
      const uint32_t offset = atomicAdd(&state->old_vec_fetch_offset, 1U);
      state->old_vector_fetch_index[offset] = i;
    }
  }
  __syncthreads();
}

// 1 block
__device__ void update_level_counts_kernel(GpuGraphState *state) {
  const int MAX_LEVEL = 128;  // TODO: move to global
  __shared__ uint32_t local_level_count[MAX_LEVEL];
  for (int i = threadIdx.x; i < MAX_LEVEL; i += blockDim.x) {
    local_level_count[i] = 0;
  }
  __syncthreads();

  int ed = min(state->cur_element_count + BATCHSZ_PER_NEW, state->max_elements);
  for (int i = state->cur_element_count + threadIdx.x; i < ed; i += blockDim.x) {
    int level = state->element_levels[i];
    for (int j = 0; j <= level; j++) {
      atomicAdd(&local_level_count[j], 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 0; i < MAX_LEVEL; i++) {
      atomicAdd(&state->level_counts[i], local_level_count[i]);
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

      for (int k = 0; k < DIM; k += kWmmaK) {
        wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_frag;
        const half *a_ptr = row_vectors + row0 * DIM + k;
        const half *b_ptr = col_vectors + col0 * DIM + k;
        wmma::load_matrix_sync(a_frag, a_ptr, DIM);
        wmma::load_matrix_sync(b_frag, b_ptr, DIM);
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
      if (i < static_cast<uint32_t>(wmma_rows) && j < wmma_cols) continue;
      float ip = 0.0f;
      const int row_st = i * DIM;
      const int col_st = j * DIM;
      for (int k = 0; k < DIM; ++k) {
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

__device__ void compute_dist_between_new_kernel(GpuGraphState *state) {
  half *new_vector_store = state->half_vector_data + state->cur_element_count * DIM;

  const uint32_t new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  if (new_count == 0) return;

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

        for (int k = 0; k < DIM; k += kWmmaK) {
          wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag;
          wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_frag;
          const half *a_ptr = new_vector_store + new_row * DIM + k;
          const half *b_ptr = new_vector_store + new_col * DIM + k;
          wmma::load_matrix_sync(a_frag, a_ptr, DIM);
          wmma::load_matrix_sync(b_frag, b_ptr, DIM);
          wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float *output_ptr = state->news_dist + new_row * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + LEVEL_SZ_THRES + new_col;
        wmma::store_matrix_sync(output_ptr, c_frag, LEVEL_SZ_THRES + BATCHSZ_PER_NEW, wmma::mem_row_major);
      }
    }
  }
  __syncthreads();

  // handle tail rows/cols not covered by WMMA full tiles
  const int wmma_rows = (new_count / 16) * 16;
  const int wmma_cols = (new_count / 16) * 16;
  for (uint32_t i = 0; i < new_count; ++i) {
    for (int j = threadIdx.x; j < new_count; j += blockDim.x) {
      if (i < static_cast<uint32_t>(wmma_rows) && j < wmma_cols) continue;
      float ip = 0.0f;
      const int row_st = i * DIM;
      const int col_st = j * DIM;
      for (int k = 0; k < DIM; ++k) {
        ip += __half2float(new_vector_store[row_st + k]) * __half2float(new_vector_store[col_st + k]);
      }
      state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + LEVEL_SZ_THRES + j] = ip;
    }
  }
  __syncthreads();

  for (uint32_t i = 0; i < new_count; ++i) {
    const uint32_t one = state->cur_element_count + i;
    for (int j = threadIdx.x; j < new_count; j += blockDim.x) {
      float ip = state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + LEVEL_SZ_THRES + j];
      state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + LEVEL_SZ_THRES + j] = -2 * ip;
      state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + LEVEL_SZ_THRES + j] += state->vector_powers[one];
      int another = state->cur_element_count + j;
      state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + LEVEL_SZ_THRES + j] += state->vector_powers[another];
      state->news_rank[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + LEVEL_SZ_THRES + j] = another;
    }
  }
}

// 1 block for 1 new vector
__device__ void connect_new_to_old_at_upper_kernel(GpuGraphState *state, int startup_lvl) {
  __shared__ uint32_t prev_neigh_rank;
  __shared__ uint32_t prev_neigh_id;
  __shared__ float prev_neighbor[DIM];
  __shared__ float prev_neigh_dist;
  __shared__ uint32_t curr_neigh_cnt;
  __shared__ uint32_t neigh_rank[BATCHSZ_PER_NEW];
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
        for (int i = 1; i < state->old_vec_fetch_offset; i++) {
          if (pruned_mask[i] == 0) {
            prev_neigh_rank = i;
            prev_neigh_id = ranked_cand[i];
            prev_neigh_dist = ranked_dist[i];
            neigh_rank[0] = prev_neigh_rank;
            curr_neigh_cnt = 1;
            pruned_mask[i] = 1;
            break;
          }
        }
      }
      __syncthreads();
      for (int i = threadIdx.x; i < DIM; i += blockDim.x) {
        prev_neighbor[i] = state->vector_data[prev_neigh_id * state->vector_dim + i];
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
        for (int i = ty; i < state->old_vec_fetch_offset; i += nrow) {
          if (i <= prev_neigh_rank) continue;
          if (pruned_mask[i] == 0) {
            can_continue = 1;
            float dist = 0.0f;
            int cand = ranked_cand[i];
            // TODO: warp-level sync
            for (int j = tx; j < DIM; j += warpSize) {
              dist += (state->vector_data[cand * state->vector_dim + j] - prev_neighbor[j]) *
                      (state->vector_data[cand * state->vector_dim + j] - prev_neighbor[j]);
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
              prev_neigh_dist = ranked_dist[i];
              neigh_rank[curr_neigh_cnt] = i;
              curr_neigh_cnt++;
              pruned_mask[i] = 1;
              break;
            }
          }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < DIM; i += blockDim.x) {
          prev_neighbor[i] = state->vector_data[prev_neigh_id * state->vector_dim + i];
        }
      }

      // Still treat as 1d block, adding all the selected edges based on the result.
      uint32_t *linkl = lv == 0 ? get_linklist0(state, vid) : get_linklist(state, vid, lv);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = lv == 0 ? (float *)get_linklist_dist0(state, vid) : (float *)get_linklist_dist(state, vid, lv);
      for (int i = threadIdx.x; i < curr_neigh_cnt; i += blockDim.x) {
        const uint32_t cand = ranked_cand[neigh_rank[i]];
        const uint32_t capacity = lv == 0 ? state->maxM0 + BATCHSZ_PER_NEW : state->M + BATCHSZ_PER_NEW;
#ifndef NDEBUG
        if (state->element_levels[cand] < lv) {
          printf("Fatal: connect_new_to_old selected off-level candidate %u at level %d for new %u\n", cand, lv, vid);
          assert(false);
        }
#endif
        int pos = atomicAdd((uint32_t *)linkl, 1);
#ifndef NDEBUG
        if (pos >= capacity) {
          printf("Fatal: new node link list out of bound at level %d for node %u\n", lv, vid);
          assert(false);
        }
#endif
        datal[pos] = cand;
        distl[pos] = ranked_dist[neigh_rank[i]];
        if (cand >= state->cur_element_count) continue;
        uint32_t *other_linkl = lv == 0 ? get_linklist0(state, cand) : get_linklist(state, cand, lv);
        auto other_datal = (uint32_t *)(other_linkl) + 1;
        float *other_distl =
            lv == 0 ? (float *)get_linklist_dist0(state, cand) : (float *)get_linklist_dist(state, cand, lv);
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
__device__ void finally_prune_for_new_kernel(GpuGraphState *state) {
  __shared__ uint32_t prev_neigh_rank;
  __shared__ uint32_t prev_neigh_id;
  __shared__ float prev_neighbor[DIM];
  __shared__ float prev_neigh_dist;
  __shared__ uint32_t curr_neigh_cnt;
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
      uint32_t *linkl = lv == 0 ? get_linklist0(state, vid) : get_linklist(state, vid, lv);
      uint32_t sz = getListCount(linkl);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = lv == 0 ? (float *)get_linklist_dist0(state, vid) : (float *)get_linklist_dist(state, vid, lv);
      if (threadIdx.x == 0) {
        shared_sz = sz;
        curr_neigh_cnt = 0;
      }
      for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        int other = datal[i];
        float dist = distl[i];
        ranked_dist[i] = dist;
        ranked_cand[i] = other;
      }
      __syncthreads();
      // static_assert(LEVEL_SZ_THRES - 32 >= BATCHSZ_PER_NEW);  // TODO: 32 is M, but for safety
      for (int i = threadIdx.x; i < new_count; i += blockDim.x) {
        if (state->element_levels[state->cur_element_count + i] >= lv) {
          uint32_t pos = atomicAdd(&shared_sz, 1);
          ranked_dist[pos] = ranked_dist[i + LEVEL_SZ_THRES];
          ranked_cand[pos] = ranked_cand[i + LEVEL_SZ_THRES];
        }
      }
      __syncthreads();
      sz = shared_sz;

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
        if (ranked_cand[i] == static_cast<uint32_t>(vid) || state->element_levels[ranked_cand[i]] < lv) {
          pruned_mask[i] = 1;
        } else {
          pruned_mask[i] = 0;
        }
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        for (int i = 0; i < sz; i++) {
          if (pruned_mask[i] == 0) {
            prev_neigh_rank = i;
            prev_neigh_id = ranked_cand[i];
            prev_neigh_dist = ranked_dist[i];
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
      for (int i = threadIdx.x; i < DIM; i += blockDim.x) {
        prev_neighbor[i] = state->vector_data[prev_neigh_id * state->vector_dim + i];
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
        for (int i = ty; i < sz; i += nrow) {
          if (i <= prev_neigh_rank) continue;
          if (pruned_mask[i] == 0) {
            can_continue = 1;
            float dist = 0.0f;
            int cand = ranked_cand[i];
            for (int j = tx; j < DIM; j += warpSize) {
              dist += (state->vector_data[cand * state->vector_dim + j] - prev_neighbor[j]) *
                      (state->vector_data[cand * state->vector_dim + j] - prev_neighbor[j]);
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
              prev_neigh_dist = ranked_dist[i];
              neigh_rank[curr_neigh_cnt] = i;
              curr_neigh_cnt++;
              pruned_mask[i] = 1;
              break;
            }
          }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < DIM; i += blockDim.x) {
          prev_neighbor[i] = state->vector_data[prev_neigh_id * state->vector_dim + i];
        }
        __syncthreads();
      }

      // Still treat as 1d block, adding all the selected edges based on the result.
      for (int i = threadIdx.x; i < curr_neigh_cnt; i += blockDim.x) {
        datal[i] = ranked_cand[neigh_rank[i]];
        distl[i] = ranked_dist[neigh_rank[i]];
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        setListCount(linkl, curr_neigh_cnt);
      }
    }
  }
}

__device__ void snapshot_frozen_link_counts_kernel(GpuGraphState *state, int max_level) {
  if (max_level < 0) return;
  max_level = min(max_level, state->maxlevel);
  const uint32_t new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  const uint32_t node_count = min(state->max_elements, state->cur_element_count + new_count);

  for (int lv = 0; lv <= max_level; ++lv) {
    for (uint32_t node_id = threadIdx.x + blockIdx.x * blockDim.x; node_id < node_count;
         node_id += blockDim.x * gridDim.x) {
      uint32_t count = 0;
      if (state->element_levels[node_id] >= lv) {
        uint32_t *linkl = lv == 0 ? get_linklist0(state, node_id) : get_linklist(state, node_id, lv);
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

// 1d grid, 1d block
__global__ void build_graph_kernel(GpuGraphState *state) {
  // build the graph with iterations: batch dependency
  cg::grid_group grid = cg::this_grid();
  for (int i = 0; i < state->max_elements; i += BATCHSZ_PER_NEW) {
    int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
    // Smallest L with level_counts[L] <= THRES (level_counts is non-increasing in L).
    int lo = -1, hi = state->maxlevel + 1;
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (state->level_counts[mid] <= LEVEL_SZ_THRES) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    const int startup_level = min(hi, state->maxlevel);
    if (blockIdx.x == 0) {
      // aggregate the old vectors available on the startup level
      aggregate_on_level_kernel(state, startup_level);  // Reset per-level changed_old_link_counts by the way.
    }
    grid.sync();
    // Replacement: compute_dist_with_old_kernel can be switched to compute_dist_block_wmma in-place per old batch.
    compute_dist_with_old_kernel(state);
    if (blockIdx.x == GRID_DIM - 1) {
      // Replacement: compute_dist_block_wmma(state, new_store, new_store, new_count, new_count, LEVEL_SZ_THRES,
      // state->cur_element_count, identity_ids)
      compute_dist_between_new_kernel(state);
    }
    grid.sync();

    // for the new vectors, sort old vectors by distance
    bitonic_sort_id_by_dis(state);
    // connect the new vectors to only the old vectors in upper levels
    connect_new_to_old_at_upper_kernel(state, startup_level);

    grid.sync();

    // connect the new vectors to the old vectors in lower levels
    snapshot_frozen_link_counts_kernel(state, startup_level - 1);
    grid.sync();
    search_knn_at_lower_kernel(state, startup_level);
    // for the new vectors, combine and sort old and new vectors by distance
    finally_prune_for_new_kernel(state);
    // Wait for every block to finish those global mutations before any block
    // starts sorting/pruning old-node adjacency, otherwise later phases can
    // observe partially updated per-level lists.
    grid.sync();

    // Sort and prune all old-node lists.
    // bitonic_sort_id_for_all_ll(state);
    // prune_neighbors_for_all_kernel(state); // Update frozen link counts for all levels.

    // Sort and prune only old-node lists that received reverse edges in this batch.
    bitonic_sort_id_for_ll(state);
    prune_neighbors_kernel(state);  // Update frozen link counts for the levels that received reverse edges.
    grid.sync();

    if (blockIdx.x == 0) {
      update_level_counts_kernel(state);
    }
    if (grid.thread_rank() == 0) {
      state->cur_element_count += new_count;
#ifndef NDEBUG
      if (state->cur_element_count % 1024 == 0) {
        printf("cur_element_count=%d\n", state->cur_element_count);
      }
#endif
    }
    grid.sync();
  }
}

cudaError_t launch_build_graph_kernel(GpuGraphState *state) {
  // TODO: consider multiple kernel launches
  void *args[] = {&state};
  dim3 gridDim(GRID_DIM);
  dim3 blockDim(BLOCK_DIM);
  cudaLaunchCooperativeKernel((void *)build_graph_kernel, gridDim, blockDim, args);
  return cudaGetLastError();
}

// 1d grid, 1d block
__device__ void compute_power_kernel(GpuGraphState *state) {
  int bid = blockIdx.x;
  if (bid >= state->max_elements) {
    return;
  }
  __shared__ float cache[DIM];
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
    curand_init(kRandomLevelSeed, static_cast<unsigned long long>(wid), 0, &rng_state);
    float sample = fmaxf(curand_uniform(&rng_state), 1.0e-7f);

    int32_t level = static_cast<int32_t>(-logf(sample) * state->mult);
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
  __shared__ __align__(32) half old_vector_store[BATCHSZ_PER_OLD * DIM];
  half *new_vector_store = state->half_vector_data + state->cur_element_count * DIM;

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
      const uint32_t old_vector_store_st = i * state->vector_dim;
      for (int j = threadIdx.x; j < DIM; j += blockDim.x) {
        old_vector_store[old_vector_store_st + j] = state->half_vector_data[vector_st + j];
      }
    }

    __syncthreads();

    // compute new-old inner products using tensor core
    {
      // TODO: need to pad the vector dimension to multiple of 16
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

          for (int k = 0; k < DIM; k += kWmmaK) {
            wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_frag;
            const half *a_ptr = new_vector_store + new_row * DIM + k;
            const half *b_ptr = old_vector_store + old_col * DIM + k;
            wmma::load_matrix_sync(a_frag, a_ptr, DIM);
            wmma::load_matrix_sync(b_frag, b_ptr, DIM);
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
    for (uint32_t i = 0; i < new_count; ++i) {
      for (int local_j = threadIdx.x; local_j < old_count; local_j += blockDim.x) {
        if (i < static_cast<uint32_t>(wmma_rows) && local_j < wmma_cols) continue;
        float ip = 0.0f;
        const int new_st = i * DIM;
        const int old_st = local_j * DIM;
        for (int k = 0; k < DIM; ++k) {
          ip += __half2float(new_vector_store[new_st + k]) * __half2float(old_vector_store[old_st + k]);
        }
        state->news_dist[i * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW) + oid + local_j] = ip;
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

// 1 block for 1 new vector
__device__ void prune_for_new_kernel(GpuGraphState *state, Neighbor *pq, int sz, int bid, int lv) {
  __shared__ uint32_t prev_neigh_rank;
  __shared__ uint32_t prev_neigh_id;
  __shared__ float prev_neighbor[DIM];
  __shared__ float prev_neigh_dist;
  __shared__ uint32_t curr_neigh_cnt;
  __shared__ uint32_t neigh_rank[BATCHSZ_PER_NEW];
  __shared__ unsigned char pruned_mask[LEVEL_SZ_THRES + BATCHSZ_PER_NEW];
  __shared__ bool can_continue;
  // connect new-to-old edges
  const int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  const int vid = state->cur_element_count + bid;

  if (state->element_levels[vid] >= lv) {
    for (int i = threadIdx.x; i < sz; i += blockDim.x) {
      pruned_mask[i] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int i = 1; i < sz; ++i) {
        if (pruned_mask[i]) continue;
        const uint32_t cand = pq[i].nodeid;
        for (int j = 0; j < i; ++j) {
          if (pq[j].nodeid == cand) {
            pruned_mask[i] = 1;
            break;
          }
        }
      }
    }
    __syncthreads();

    uint32_t *linkl = lv == 0 ? get_linklist0(state, vid) : get_linklist(state, vid, lv);
    uint32_t *datal = (uint32_t *)(linkl + 1);
    float *distl = lv == 0 ? (float *)get_linklist_dist0(state, vid) : (float *)get_linklist_dist(state, vid, lv);
    if (threadIdx.x == 0) {
      prev_neigh_rank = 0;
      prev_neigh_id = pq[0].nodeid;
      prev_neigh_dist = pq[0].distance;
      datal[0] = prev_neigh_id;
      distl[0] = prev_neigh_dist;
      neigh_rank[0] = prev_neigh_rank;
      curr_neigh_cnt = 1;
      pruned_mask[0] = 1;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < DIM; i += blockDim.x) {
      prev_neighbor[i] = state->vector_data[prev_neigh_id * state->vector_dim + i];
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
      for (int i = ty; i < sz; i += nrow) {
        if (i <= prev_neigh_rank) continue;
        if (pruned_mask[i] == 0) {
          can_continue = 1;
          float dist = 0.0f;
          int cand = pq[i].nodeid;
          // TODO: warp-level sync
          for (int j = tx; j < DIM; j += warpSize) {
            dist += (state->vector_data[cand * state->vector_dim + j] - prev_neighbor[j]) *
                    (state->vector_data[cand * state->vector_dim + j] - prev_neighbor[j]);
          }
          for (int lane = warpSize / 2; lane > 0; lane /= 2) {
            dist += __shfl_down_sync(0xffffffff, dist, lane);
          }
          if (tx == 0 && dist < prev_neigh_dist) {
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
            prev_neigh_id = pq[i].nodeid;
            prev_neigh_dist = pq[i].distance;
            neigh_rank[curr_neigh_cnt] = i;
            datal[curr_neigh_cnt] = prev_neigh_id;
            distl[curr_neigh_cnt] = prev_neigh_dist;
            curr_neigh_cnt++;
            break;
          }
        }
      }
      __syncthreads();
      for (int i = threadIdx.x; i < DIM; i += blockDim.x) {
        prev_neighbor[i] = state->vector_data[prev_neigh_id * state->vector_dim + i];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      setListCount(linkl, curr_neigh_cnt);
    }
  }
}

// 1 block for 1 (old) vector
// assume that neighbor list has been sorted
__device__ void prune_neighbors_kernel(GpuGraphState *state) {
  __shared__ uint32_t prev_neigh_rank;
  __shared__ uint32_t prev_neigh_id;
  __shared__ float prev_neigh[DIM];
  __shared__ float prev_neigh_dist;
  __shared__ uint32_t curr_neigh_cnt;
  __shared__ unsigned char pruned_mask[BATCHSZ_PER_OLD + BATCHSZ_PER_NEW];
  __shared__ bool can_continue;

  for (int lv = 0; lv <= state->maxlevel; ++lv) {
    const uint32_t count = state->changed_old_link_counts[lv];
    const uint32_t base = changed_old_link_level_offset(state, lv);
    for (uint32_t idx = blockIdx.x; idx < count; idx += gridDim.x) {
      const uint32_t vid = state->changed_old_links[base + idx];
      int M = lv ? state->M : state->maxM0;

      uint32_t *linkl = lv == 0 ? get_linklist0(state, vid) : get_linklist(state, vid, lv);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = lv == 0 ? (float *)get_linklist_dist0(state, vid) : (float *)get_linklist_dist(state, vid, lv);
      int sz = getListCount(linkl);

      if (sz > M) {
        if (threadIdx.x == 0) {
          prev_neigh_rank = 0;
          prev_neigh_id = datal[0];
          prev_neigh_dist = distl[0];
          datal[0] = prev_neigh_id;
          distl[0] = prev_neigh_dist;
          curr_neigh_cnt = 1;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < sz; i += blockDim.x) {
          pruned_mask[i] = 0;
        }
        for (int i = threadIdx.x; i < state->vector_dim; i += blockDim.x) {
          prev_neigh[i] = state->vector_data[prev_neigh_id * state->vector_dim + i];
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

          for (int i = ty; i < sz; i += nrow) {
            if (i <= prev_neigh_rank) continue;
            if (pruned_mask[i] == 0) {
              can_continue = 1;
              float dist = 0.0f;
              const int cand = datal[i];
              for (int j = tx; j < DIM; j += warpSize) {
                dist += (state->vector_data[cand * state->vector_dim + j] - prev_neigh[j]) *
                        (state->vector_data[cand * state->vector_dim + j] - prev_neigh[j]);
              }
              for (int lane = warpSize / 2; lane > 0; lane /= 2) {
                dist += __shfl_down_sync(0xffffffff, dist, lane);
              }
              if (tx == 0 && dist < prev_neigh_dist) {
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
                prev_neigh_dist = distl[i];
                distl[curr_neigh_cnt] = prev_neigh_dist;
                datal[curr_neigh_cnt] = prev_neigh_id;
                curr_neigh_cnt++;
                pruned_mask[i] = 1;
                break;
              }
            }
          }
          __syncthreads();
          for (int i = threadIdx.x; i < state->vector_dim; i += blockDim.x) {
            prev_neigh[i] = state->vector_data[prev_neigh_id * state->vector_dim + i];
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

__device__ void prune_neighbors_for_all_kernel(GpuGraphState *state) {
  __shared__ uint32_t prev_neigh_rank;
  __shared__ uint32_t prev_neigh_id;
  __shared__ float prev_neigh[DIM];
  __shared__ float prev_neigh_dist;
  __shared__ uint32_t curr_neigh_cnt;
  __shared__ unsigned char pruned_mask[BATCHSZ_PER_OLD + BATCHSZ_PER_NEW];
  __shared__ bool can_continue;

  for (int vid = blockIdx.x; vid < state->cur_element_count; vid += gridDim.x) {
    for (int lv = 0; lv <= state->element_levels[vid]; ++lv) {
      int M = lv ? state->M : state->maxM0;

      uint32_t *linkl = lv == 0 ? get_linklist0(state, vid) : get_linklist(state, vid, lv);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = lv == 0 ? (float *)get_linklist_dist0(state, vid) : (float *)get_linklist_dist(state, vid, lv);
      int sz = getListCount(linkl);
      if (sz == get_frozen_link_count(state, vid, lv)) continue;

      if (sz > M) {
        if (threadIdx.x == 0) {
          prev_neigh_rank = 0;
          prev_neigh_id = datal[0];
          prev_neigh_dist = distl[0];
          datal[0] = prev_neigh_id;
          distl[0] = prev_neigh_dist;
          curr_neigh_cnt = 1;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < sz; i += blockDim.x) {
          pruned_mask[i] = 0;
        }
        for (int i = threadIdx.x; i < state->vector_dim; i += blockDim.x) {
          prev_neigh[i] = state->vector_data[prev_neigh_id * state->vector_dim + i];
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

          for (int i = ty; i < sz; i += nrow) {
            if (i <= prev_neigh_rank) continue;
            if (pruned_mask[i] == 0) {
              can_continue = 1;
              float dist = 0.0f;
              const int cand = datal[i];
              for (int j = tx; j < DIM; j += warpSize) {
                dist += (state->vector_data[cand * state->vector_dim + j] - prev_neigh[j]) *
                        (state->vector_data[cand * state->vector_dim + j] - prev_neigh[j]);
              }
              for (int lane = warpSize / 2; lane > 0; lane /= 2) {
                dist += __shfl_down_sync(0xffffffff, dist, lane);
              }
              if (tx == 0 && dist < prev_neigh_dist) {
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
                prev_neigh_dist = distl[i];
                distl[curr_neigh_cnt] = prev_neigh_dist;
                datal[curr_neigh_cnt] = prev_neigh_id;
                curr_neigh_cnt++;
                pruned_mask[i] = 1;
                break;
              }
            }
          }
          __syncthreads();
          for (int i = threadIdx.x; i < state->vector_dim; i += blockDim.x) {
            prev_neigh[i] = state->vector_data[prev_neigh_id * state->vector_dim + i];
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
  const int new_count = min(BATCHSZ_PER_NEW, state->max_elements - state->cur_element_count);
  for (int bid = blockIdx.x; bid < new_count; bid += gridDim.x) {
    const int vid = state->cur_element_count + bid;
    const int lvl = state->element_levels[vid];
    const int tx = threadIdx.x % warpSize;
    const int ty = threadIdx.x / warpSize;
    const int nrow = blockDim.x / warpSize;

    // Find the entry point
    if (threadIdx.x == 0) {
      auto ranks = state->news_rank + bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
      auto dists = state->news_dist + bid * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
      // for (int i = 0; i < state->old_vec_fetch_offset; i++) {
      //   if (ranks[i] < state->cur_element_count) {
      //     curr_obj_shared = ranks[i];
      //     curr_dist_bits_shared = __float_as_int(dists[i]);
      //     break;
      //   }
      // }
      curr_obj_shared = ranks[0];
      curr_dist_bits_shared = __float_as_int(dists[0]);
      visited = state->visited + blockIdx.x * state->max_elements;
      if (bid == blockIdx.x) {
        visited_tag = 0;
      }
    }
    __syncthreads();
    int lv = startup_lv - 1;
    while (lv > lvl) {
      if (threadIdx.x == 0) {
        changed = 1;
      }
      __syncthreads();
      while (1) {
        if (threadIdx.x == 0) {
          changed = 0;
        }
        __syncthreads();

        const int size = get_frozen_link_count(state, curr_obj_shared, lv);
        uint32_t *linkl = size > 0 ? get_linklist(state, curr_obj_shared, lv) : nullptr;
        uint32_t *datal = size > 0 ? (uint32_t *)(linkl + 1) : nullptr;

        for (int i = ty; i < size; i += nrow) {
          uint32_t cand = datal[i];
          float dist = 0.0f;
          for (int j = tx; j < DIM; j += warpSize) {
            dist +=
                (state->vector_data[cand * state->vector_dim + j] - state->vector_data[vid * state->vector_dim + j]) *
                (state->vector_data[cand * state->vector_dim + j] - state->vector_data[vid * state->vector_dim + j]);
          }
          for (int lane = warpSize / 2; lane > 0; lane /= 2) {
            dist += __shfl_down_sync(0xffffffff, dist, lane);
          }
          if (tx == 0) {
            int dist_bits = __float_as_int(dist);
            int old_best = atomicMin(&curr_dist_bits_shared, dist_bits);  // positive float trick
            if (dist_bits < old_best) {
              curr_obj_shared = cand;
              atomicExch(&changed, 1);
            }
          }
        }
        __syncthreads();
        if (!changed) {
          break;
        }
      }
      lv--;
      __syncthreads();
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
#ifndef NDEBUG
        printf("Run to here: %s %d, Vector %d, Level %d\n", __FILE__, __LINE__, vid, lv);
#endif
        ++visited_tag;
      }
      __syncthreads();

      // Better not to do the beam search
      while (candq_sz > 0) {
        __syncthreads();
        if (topq_sz >= EFC && candq[0].distance > topq_max) {
          break;
        }
        __syncthreads();

        const int tx = threadIdx.x % warpSize;
        const int ty = threadIdx.x / warpSize;
        const int nrow = blockDim.x / warpSize;

        Neighbor tmp{candq[0].distance, candq[0].nodeid, candq[0].checked};
        if (threadIdx.x == 0) {
          atomicExch(&visited[tmp.nodeid], visited_tag);
          candq[0] = candq[candq_sz - 1];
          candq_sz--;
        }
        __syncthreads();
        const int size = get_frozen_link_count(state, tmp.nodeid, lv);
        uint32_t *linkl =
            size > 0 ? (lv == 0 ? get_linklist0(state, tmp.nodeid) : get_linklist(state, tmp.nodeid, lv)) : nullptr;
        uint32_t *datal = size > 0 ? (uint32_t *)(linkl + 1) : nullptr;
        for (int i = ty; i < size; i += nrow) {  // compute the neighbors at the same time
          uint32_t cand = datal[i];
#ifndef NDEBUG
          if (state->element_levels[cand] < lv) {
            printf("Fatal: found off-level candidate %u at level %d from node %u\n", cand, lv, tmp.nodeid);
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
          for (int j = tx; j < DIM; j += warpSize) {
            dist +=
                (state->vector_data[cand * state->vector_dim + j] - state->vector_data[vid * state->vector_dim + j]) *
                (state->vector_data[cand * state->vector_dim + j] - state->vector_data[vid * state->vector_dim + j]);
          }
          for (int lane = warpSize / 2; lane > 0; lane /= 2) {
            dist += __shfl_down_sync(0xffffffff, dist, lane);
          }
          // Instead of using lock, use atomic.
          if (tx == 0) {
            if (topq_sz < EFC || dist < topq_max) {  // TODO: Multiple threads get here at the same time
              int topq_pos = atomicAdd(&topq_sz, 1);
              if (topq_pos < TOPQ_SZ) {
                topq[topq_pos].nodeid = cand;
                topq[topq_pos].distance = dist;
              }
              int candq_pos = atomicAdd(&candq_sz, 1);
              if (candq_pos < CANDQ_SZ) {
                candq[candq_pos].nodeid = cand;
                candq[candq_pos].distance = dist;
              }
            }
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          topq_sz = min(topq_sz, TOPQ_SZ);
          candq_sz = min(candq_sz, CANDQ_SZ);
        }
        __syncthreads();
        bitonic_sort_pq(topq, topq_sz);
        bitonic_sort_pq(candq, candq_sz);
        if (threadIdx.x == 0) {
          topq_sz = min(topq_sz, EFC);
          topq_max = topq_sz > 0 ? topq[topq_sz - 1].distance : INFINITY;
        }
        __syncthreads();
      }
      __syncthreads();
      // prune the old vectors for the new
      prune_for_new_kernel(state, topq, topq_sz, bid, lv);

      uint32_t *linkl = lv == 0 ? get_linklist0(state, vid) : get_linklist(state, vid, lv);
      int sz = getListCount(linkl);
      uint32_t *datal = (uint32_t *)(linkl + 1);
      float *distl = lv == 0 ? (float *)get_linklist_dist0(state, vid) : (float *)get_linklist_dist(state, vid, lv);
      // update the current object and distance
      if (threadIdx.x == 0) {
        curr_obj_shared = datal[0];
        curr_dist_bits_shared = __float_as_int(distl[0]);
      }
      __syncthreads();
      // add reverse edges for the old vectors
      for (int i = threadIdx.x; i < sz; i += blockDim.x) {
        uint32_t *other_linkl = lv == 0 ? get_linklist0(state, datal[i]) : get_linklist(state, datal[i], lv);
        uint32_t *other_datal = (uint32_t *)(other_linkl) + 1;
        float *other_distl =
            lv == 0 ? (float *)get_linklist_dist0(state, datal[i]) : (float *)get_linklist_dist(state, datal[i], lv);
        uint32_t pos = atomicAdd((uint32_t *)(other_linkl), 1);
        record_changed_old_link(state, datal[i], lv);
        const uint32_t capacity = lv == 0 ? state->maxM0 + BATCHSZ_PER_NEW : state->M + BATCHSZ_PER_NEW;
#ifndef NDEBUG
        if (pos >= capacity) {
          printf("Fatal: link list out of bound\n");
          assert(false);
        }
#endif
        other_datal[pos] = vid;
        other_distl[pos] = distl[i];
      }

      lv--;
      __syncthreads();
    }
  }
}
