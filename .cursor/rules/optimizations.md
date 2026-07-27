- [x] Instead of doing the `bitonic_sort_id_for_ll` and `prune_neighbors_kernel` inside `build_graph_kernel` for all the old vectors, we should store and handle only the old vectors whose list count have changed on each level during current batch of insertion (just like the treatment to old_vec_fetch_offset). This should save time compared to iterating over the whole old vectors, even though there is logic handling unchanged neighbor list for old vectors now.

- [x] Parallelize `aggregate_on_level_kernel` scan across the grid. Today only block 0 (64 threads) scans all `cur_element_count` elements to build `old_vector_fetch_index`, using `atomicAdd` on `old_vec_fetch_offset` for compaction. That is reasonable for the bounded output (<= `LEVEL_SZ_THRES` ~= 1024, enforced by `startup_level` selection) and avoids cross-block atomic contention on a single counter, but it does not scale as `cur_element_count` grows - the full scan dominates long before the atomics do. A better approach: let each block scan a chunk of old vectors, produce a block-local match list/count, then prefix-sum block counts and scatter into the global `old_vector_fetch_index` (or use block-local shared-memory compaction within each chunk). Within a single block, `atomicAdd` could also be replaced by a two-pass prefix-sum compaction, but parallelizing the scan is the main win. (`src/impl/kernel.cu`:320, `src/impl/kernel.cu`:329, `src/impl/kernel.cu`:929)

## `search_knn_at_lower_kernel` bottlenecks and optimization points

- [x] Cache the query vector (`vid`) in shared memory once per insertion (`bid`) and reuse it across all neighbor distance evaluations. (`src/impl/kernel.cu`:1495, `src/impl/kernel.cu`:1512, `src/impl/kernel.cu`:1548, `src/impl/kernel.cu`:1630)

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
