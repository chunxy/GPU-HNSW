#pragma once

#include <cstdint>

enum BuildPhase : int {
  kBuildPhaseAggregate = 0,
  kBuildPhaseDistOldNew,
  kBuildPhaseDistNewNew,
  kBuildPhaseSortOldByDist,
  kBuildPhaseConnectUpper,
  kBuildPhaseSnapshotFrozen,
  kBuildPhaseSearchPruneLower,
  kBuildPhaseCombinePruneUpper,
  kBuildPhaseAddReverse,
  kBuildPhaseSortPruneOld,
  kBuildPhaseUpdateBatch,
  kBuildPhaseCount,
};

#ifdef PROFILE_BUILD_PHASES

inline const char *build_phase_name(BuildPhase phase) {
  switch (phase) {
    case kBuildPhaseAggregate:
      return "aggregate_on_level";
    case kBuildPhaseDistOldNew:
      return "dist_old_new";
    case kBuildPhaseDistNewNew:
      return "dist_new_new";
    case kBuildPhaseSortOldByDist:
      return "sort_old_by_dist";
    case kBuildPhaseConnectUpper:
      return "connect_upper";
    case kBuildPhaseSnapshotFrozen:
      return "snapshot_frozen";
    case kBuildPhaseSearchPruneLower:
      return "search_prune_lower";
    case kBuildPhaseCombinePruneUpper:
      return "combine_prune_upper";
    case kBuildPhaseAddReverse:
      return "add_reverse";
    case kBuildPhaseSortPruneOld:
      return "sort_prune_old";
    case kBuildPhaseUpdateBatch:
      return "update_batch";
    default:
      return "unknown";
  }
}

void print_build_phase_profile(const uint64_t *cycles, uint64_t batch_count);

void print_search_block_profile(
    const uint64_t *block_cycles,
    const uint64_t *block_clear_cycles,
    const uint64_t *block_expands,
    uint64_t batch_count);

#endif  // PROFILE_BUILD_PHASES
