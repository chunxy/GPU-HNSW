const int MAX_VISITED = 1000000;

#ifndef NDEBUG
const int GRID_DIM = 1024;
#else
const int GRID_DIM = 1024;
#endif

const int BLOCK_DIM = 128;

const int BATCHSZ_PER_NEW = 2048; // Limited by the HBM SIZE

const int BATCHSZ_PER_OLD = 32;

const int LEVEL_SZ_THRES = 2048;

// Largest vector dimension supported by GPU build kernels (sizes shared-memory vector buffers).
const int MAX_DIM = 1024;

const int MAX_HNSW_LEVEL = 32; // The element level is in [0, MAX_HNSW_LEVEL)

const int MAX_M0 = 32;

// Largest ef_construction supported by GPU build kernels (sizes shared-memory top-q buffers).
const int MAX_EFC = 100;

const int TOPQ_SZ = MAX_EFC + MAX_M0;  // MAX_M0 for the max number of neighbors

// Build-time link lists reserve maxM0 + REVERSE_HEADROOM (L0) or
// M + REVERSE_HEADROOM slots
// for reverse-edge headroom on old nodes. Search no longer dumps TOPQ into the list.
const int REVERSE_HEADROOM = 0;

const int CANDQ_SZ = 200;

// search_knn_at_lower_kernel: per-warp neighbor staging during beam search
const int SEARCH_WARP_COUNT = BLOCK_DIM / 32;
const int WARP_STAGING_CAP = 16;  // >= ceil(maxM0 / SEARCH_WARP_COUNT); maxM0 is typically 2 * M

// static_assert(BATCHSZ_PER_NEW > GRID_DIM);
// static_assert(BATCHSZ_PER_OLD * GRID_DIM <= LEVEL_SZ_THRES);
static_assert(BATCHSZ_PER_NEW % 16 == 0);  // ensure WMMA
static_assert(BATCHSZ_PER_OLD % 16 == 0);  // ensure WMMA
static_assert(MAX_DIM % 16 == 0);
static_assert(BLOCK_DIM % 32 == 0);
static_assert(TOPQ_SZ <= LEVEL_SZ_THRES, "TOPQ must fit in the per-new news_* prefix");