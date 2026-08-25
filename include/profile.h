#pragma once

#include <cstdint>

enum BuildPhase : int {
  kBuildPhaseAggregate = 0,
  kBuildPhaseDistOldNew,
  kBuildPhaseLoadNewNew,
  kBuildPhaseSortOldByDist,
  kBuildPhaseConnectUpper,
  kBuildPhaseSnapshotFrozen,
  kBuildPhaseSearchLower,
  kBuildPhaseCombinePruneNew,
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
    case kBuildPhaseLoadNewNew:
      return "load_new_new";
    case kBuildPhaseSortOldByDist:
      return "sort_old_by_dist";
    case kBuildPhaseConnectUpper:
      return "connect_upper";
    case kBuildPhaseSnapshotFrozen:
      return "snapshot_frozen";
    case kBuildPhaseSearchLower:
      return "search_lower";
    case kBuildPhaseCombinePruneNew:
      return "finally_prune_new";
    case kBuildPhaseSortPruneOld:
      return "sort_prune_old";
    case kBuildPhaseUpdateBatch:
      return "update_batch";
    default:
      return "unknown";
  }
}

void print_build_phase_profile(const uint64_t *cycles, uint64_t batch_count);

#endif  // PROFILE_BUILD_PHASES
