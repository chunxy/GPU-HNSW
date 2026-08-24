- [x] Instead of doing the `bitonic_sort_id_for_ll` and `prune_neighbors_kernel` inside `build_graph_kernel` for all the old vectors, we should store and handle only the old vectors whose list count have changed on each level during current batch of insertion (just like the treatment to old_vec_fetch_offset). This should save time compared to iterating over the whole old vectors, even though there is logic handling unchanged neighbor list for old vectors now.

- [x] Parallelize `aggregate_on_level_kernel` scan across the grid. Today only block 0 (64 threads) scans all `cur_element_count` elements to build `old_vector_fetch_index`, using `atomicAdd` on `old_vec_fetch_offset` for compaction. That is reasonable for the bounded output (<= `LEVEL_SZ_THRES` ~= 1024, enforced by `startup_level` selection) and avoids cross-block atomic contention on a single counter, but it does not scale as `cur_element_count` grows - the full scan dominates long before the atomics do. A better approach: let each block scan a chunk of old vectors, produce a block-local match list/count, then prefix-sum block counts and scatter into the global `old_vector_fetch_index` (or use block-local shared-memory compaction within each chunk). Within a single block, `atomicAdd` could also be replaced by a two-pass prefix-sum compaction, but parallelizing the scan is the main win. (`src/impl/kernel.cu`:320, `src/impl/kernel.cu`:329, `src/impl/kernel.cu`:929)

## `compute_dist_with_old_kernel` bottlenecks and optimization points

- [ ] Fuse the inner-product → L2 epilogue into the same warp/thread that produced the IP, instead of a second block-wide loop over `news_dist`. Two legal shapes: (1) store full IP, `__syncthreads()`, convert — required because the convert pass reads other warps' stores; (2) after the K-reduction finishes for a WMMA 16×16 tile (and for each tail `(i,j)`), apply `dist = ||x||^2 + ||y||^2 - 2*ip` and write L2 + `news_rank` locally (`__syncwarp()` only if the tile round-trips through memory). Prefer (2): one global store, no mid-kernel block barrier, standard GEMM epilogue. "Partial" means a slice of the **output** tile, not converting each K-slice (that would add norms once per K-tile). Conversion index is global column `j` in `[oid, oid + old_count)`. (`src/impl/kernel.cu`:1164, `src/impl/kernel.cu`:1209, `src/impl/kernel.cu`:1220, `src/impl/kernel.cu`:1228, `src/impl/kernel.cu`:1242, `src/impl/kernel.cu`:1244)

- [ ] Keep `old_vector_store` in SMEM unless occupancy measurement says SMEM is the limiter after raising `GRID_DIM`. Packing scattered `old_vector_fetch_index` rows into a dense 16-wide B panel is required for WMMA; the home of that panel (SMEM vs HBM) is optional. 32 KiB does not block more blocks at `GRID_DIM = 32`; `BATCHSZ_PER_OLD * GRID_DIM <= LEVEL_SZ_THRES` already allows up to 256 blocks with this buffer. The fat cooperative `build_graph_kernel` is more often register-bound than SMEM-bound. (`src/impl/kernel.cu`:1165, `src/impl/kernel.cu`:1176, `include/param.h`:13, `include/param.h`:38)

- [ ] Parallelize the old-vector gather across warps (disjoint `i * MAX_DIM` slices) instead of loading one vector with the whole block. That does **not** remove the gather `__syncthreads()` while every warp still MMA-reads the shared 16-col B panel. Dropping the block barrier requires warp-private B (16 old columns per warp, or `load_matrix_sync` B from packed HBM). (`src/impl/kernel.cu`:1176, `src/impl/kernel.cu`:1184)

- [ ] Tail: `BATCHSZ_* % 16 == 0` is not enough; remainders are `new_count` / `old_count`. Skip the scalar loop when both are multiples of 16; otherwise only walk leftover rows/cols (or pad to 16 and delete the tail). (`src/impl/kernel.cu`:1225, `include/param.h`:39)

## `search_knn_at_lower_kernel` bottlenecks and optimization points

- [x] Do not keep the query vector in `__shared__ float query_vec[MAX_DIM]`. It already lives in `vector_data[vid * dim]`; a pointer plus L2 reuse is enough. The 4 KiB SMEM reservation does not raise occupancy under the current cooperative `GRID_DIM`. (`src/impl/kernel.cu`:1458)

- [x] Replace per-iteration full `bitonic_sort_pq(topq, ...)` and `bitonic_sort_pq(candq, ...)` with per-warp staging + thread-0 batch merge via `MinPqPush`/`MaxPqPush`. One `bitonic_sort_pq(topq)` remains per level before `prune_for_new_kernel` (prune expects sorted-by-distance input). (`src/impl/kernel.cu`:201, `src/impl/kernel.cu`:1638, `src/impl/kernel.cu`:1685, `src/impl/kernel.cu`:1691)

  **Implemented approach:** Each warp stages unvisited neighbors in `warp_staging[ty][]` (lane 0 only, no admission race). After one `__syncthreads()`, thread 0 calls `merge_staged_neighbors_into_queues` (admission + heap push + `MaxPqPop` when `topq_sz > EFC`). `MinPqPop` replaces manual `candq[0]` removal to keep the min-heap valid without per-step `bitonic_sort_pq(candq)`.

- [x] Reduce synchronization pressure in the beam-search loop. There are multiple `__syncthreads()` calls per expansion step; some can be consolidated, especially around thread-0-only queue checks and metadata updates. (`src/impl/kernel.cu`:1625, `src/impl/kernel.cu`:1629, `src/impl/kernel.cu`:1640, `src/impl/kernel.cu`:1683, `src/impl/kernel.cu`:1688)

- [ ] Reduce global atomic traffic on visited checks (`atomicExch(&visited[cand], visited_tag)`) by batching/warp-coalescing candidate handling or using a cheaper visited-marking scheme where safe. (`src/impl/kernel.cu`:1637, `src/impl/kernel.cu`:1658)

- [x] Fix the queue insertion race/oversubscription path (`topq_sz`/`candq_sz` with `atomicAdd` then post-clamp). Replaced by per-warp staging + serial merge on thread 0. (`src/impl/kernel.cu`:201, `src/impl/kernel.cu`:1673, `src/impl/kernel.cu`:1685)

- [x] Optimize upper-level greedy descent (`while (lv > lvl)`) where `atomicMin(&curr_dist_bits_shared, ...)` and repeated distance computations can cause contention and extra memory traffic. (`src/impl/kernel.cu`:1529, `src/impl/kernel.cu`:1547, `src/impl/kernel.cu`:1557)

- [x] Slim `prune_for_new_kernel` work inside `search_lower` phase. It currently includes thread-0-heavy duplicate checks and iterative pruning with more distance evaluations and barriers. (`src/impl/kernel.cu`:1169, `src/impl/kernel.cu`:1187, `src/impl/kernel.cu`:1231, `src/impl/kernel.cu`:1694)

- [ ] Evaluate reverse-edge insertion overhead (`atomicAdd` on `other_linkl` count + `record_changed_old_link`). The scattered atomics/writes can serialize on high-degree hubs; consider staged buffering then merge. (`src/impl/kernel.cu`:1687, `src/impl/kernel.cu`:1688, `src/impl/kernel.cu`:116)

- [ ] Increase effective parallelism per new vector path (or reduce serial work per block). Current launch geometry (`GRID_DIM=32`, `BLOCK_DIM=64`, `BATCHSZ_PER_NEW=256`) means each block handles multiple vectors serially (`bid += gridDim.x`). (`include/param.h`:4, `include/param.h`:9, `include/param.h`:11, `src/impl/kernel.cu`:1505)

- [ ] Reconcile GPU compile-time `EFC` (`include/param.h`) with runtime CLI `--efc` expectation used in benchmark commands. Mismatch can force wider search than intended and inflate `search_lower` time. (`include/param.h`:21, `src/impl/kernel.cu`:1592, `src/impl/kernel.cu`:1662, `src/benchs/bench-build-gpu.cpp`:41)
