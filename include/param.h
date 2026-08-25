const int MAX_VISITED = 1000000;

#ifndef NDEBUG
const int GRID_DIM = 16;
#else
const int GRID_DIM = 32;
#endif

const int BLOCK_DIM = 256;

const int BATCHSZ_PER_NEW = 512;

const int BATCHSZ_PER_OLD = 16;

const int LEVEL_SZ_THRES = 4096;

// Largest vector dimension supported by GPU build kernels (sizes shared-memory vector buffers).
const int MAX_DIM = 1024;

const int MAX_HNSW_LEVEL = 32;

const int MAX_M0 = 32;

// Largest ef_construction supported by GPU build kernels (sizes shared-memory top-q buffers).
const int MAX_EFC = 50;

const int TOPQ_SZ = MAX_EFC + 16;  // 16 for the max number of neighbors

// Build-time link lists reserve maxM0 + BATCHSZ_PER_NEW (L0) or M + BATCHSZ_PER_NEW slots.
// Lower-level search stages raw beam-search results there before finally_prune.
const int MAX_LINK_M0_HEADROOM = 128;  // compile-time bound for maxM0 (= 2 * M)

const int CANDQ_SZ = 200;

// search_knn_at_lower_kernel: per-warp neighbor staging during beam search
const int SEARCH_WARP_COUNT = BLOCK_DIM / 32;
const int WARP_STAGING_CAP = 16;  // >= ceil(maxM0 / SEARCH_WARP_COUNT); maxM0 is typically 2 * M

static_assert(BATCHSZ_PER_NEW > GRID_DIM);
static_assert(BATCHSZ_PER_OLD * GRID_DIM <= LEVEL_SZ_THRES);
static_assert(BATCHSZ_PER_NEW % 16 == 0); // ensure WMMA
static_assert(BATCHSZ_PER_OLD % 16 == 0); // ensure WMMA
static_assert(MAX_DIM % 16 == 0);
static_assert(BLOCK_DIM % 32 == 0);
static_assert(TOPQ_SZ <= BATCHSZ_PER_NEW + MAX_LINK_M0_HEADROOM,
              "TOPQ_SZ must fit in build-time link list capacity (maxM0 + BATCHSZ_PER_NEW)");