const int MAX_VISITED = 1000000;

#ifndef NDEBUG
const int GRID_DIM = 32;
#else
const int GRID_DIM = 32;
#endif

const int BLOCK_DIM = 64;

const int BATCHSZ_PER_NEW = 256;

const int BATCHSZ_PER_OLD = 32;

const int LEVEL_SZ_THRES = 1024;

const int DIM = 128;

const int MAX_HNSW_LEVEL = 32;

const int EFC = 50;  // TODO: move to global

const int TOPQ_SZ = EFC + 16;  // 16 for the max number of neighbors

const int CANDQ_SZ = 200;

static_assert(BATCHSZ_PER_NEW > GRID_DIM);
static_assert(BATCHSZ_PER_OLD * GRID_DIM <= LEVEL_SZ_THRES);
static_assert(DIM % 16 == 0);