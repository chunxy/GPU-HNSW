const int MAX_VISITED = 1000000;

#ifndef NDEBUG
const int GRID_DIM = 32;
#else
const int GRID_DIM = 32;
#endif

const int BLOCK_DIM = 128;

const int BATCHSZ_PER_NEW = 256;

const int BATCHSZ_PER_OLD = 32;

const int LEVEL_SZ_THRES = 1024;

const int DIM = 128;

const int MAX_HNSW_LEVEL = 32;

const int EFC = 50;  // TODO: move to global

const int TOPQ_SZ = EFC + 16;  // 16 for the max number of neighbors

// Build-time link lists reserve maxM0 + BATCHSZ_PER_NEW (L0) or M + BATCHSZ_PER_NEW slots.
// Lower-level search stages raw beam-search results there before finally_prune.
const int MAX_LINK_M0_HEADROOM = 128;  // compile-time bound for maxM0 (= 2 * M)

const int CANDQ_SZ = 200;

// search_knn_at_lower_kernel: per-warp neighbor staging during beam search
const int SEARCH_WARP_COUNT = BLOCK_DIM / 32;
const int WARP_STAGING_CAP = 16;  // >= ceil(maxM0 / SEARCH_WARP_COUNT); maxM0 is typically 2 * M

static_assert(BATCHSZ_PER_NEW > GRID_DIM);
static_assert(BATCHSZ_PER_OLD * GRID_DIM <= LEVEL_SZ_THRES);
static_assert(DIM % 16 == 0);
static_assert(BLOCK_DIM % 32 == 0);
static_assert(TOPQ_SZ <= BATCHSZ_PER_NEW + MAX_LINK_M0_HEADROOM,
              "TOPQ_SZ must fit in build-time link list capacity (maxM0 + BATCHSZ_PER_NEW)");