#include <fmt/chrono.h>
#include <fmt/format.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>
#include "config.h"
#include "hnswlib/hnswlib.h"
#include "json.hpp"
#include "utils/Pod.h"
#include "utils/funcs.h"

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  Args args(argc, argv);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[args.datacard];

  if (args.k > c.n_groundtruth) {
    fmt::print("k({}) is greater than the number of groundtruth({}). Exit.\n", args.k, c.n_groundtruth);
    exit(-1);
  }

  float *xb, *xq;
  load_base_and_query(c, xb, xq);

  const string method = "CPU-single-upper-dist";
  string workload = fmt::format(fmt::runtime(WORKLOAD_TMPL), args.datacard, args.k);
  string build = fmt::format("M_{}_efc_{}", args.M, args.efc);

  hnswlib::L2Space space(c.vector_dim);
  hnswlib::HierarchicalNSW<float> index(&space, c.n_base, args.M, args.efc);
#ifdef HNSW_COUNT_UPPER_LEVEL_DIST
  index.resetDistCounts();
#endif

  fmt::print(
      "Counting upper-level (layer > 0) distance computations on {} (n={}, dim={}, M={}, efc={})\n",
      c.name,
      c.n_base,
      c.vector_dim,
      args.M,
      args.efc);

  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < c.n_base; ++i) {
    index.addPoint(xb + i * c.vector_dim, i);
    if ((i + 1) % 100000 == 0) {
#ifdef HNSW_COUNT_UPPER_LEVEL_DIST
      fmt::print(
          "  inserted {} / {}  upper-level dists={}  total dists={}\n",
          i + 1,
          c.n_base,
          hnswlib::HierarchicalNSW<float>::getUpperLevelDistCount(),
          hnswlib::HierarchicalNSW<float>::getTotalDistCount());
#else
      fmt::print("  inserted {} / {}\n", i + 1, c.n_base);
#endif
    }
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = end - start;

#ifdef HNSW_COUNT_UPPER_LEVEL_DIST
  const uint64_t upper = hnswlib::HierarchicalNSW<float>::getUpperLevelDistCount();
  const uint64_t total = hnswlib::HierarchicalNSW<float>::getTotalDistCount();
#else
  const uint64_t upper = 0;
  const uint64_t total = 0;
#endif
  const double frac = total > 0 ? static_cast<double>(upper) / static_cast<double>(total) : 0.0;

  const int max_level = index.maxlevel_;
  std::vector<uint64_t> assigned_to_level(max_level + 1, 0);
  const size_t n_inserted = index.cur_element_count;
  for (size_t i = 0; i < n_inserted; ++i) {
    const int lvl = index.element_levels_[i];
    if (lvl >= 0 && lvl <= max_level) {
      ++assigned_to_level[lvl];
    }
  }
  // A node assigned to level L also exists on every layer 0..L.
  std::vector<uint64_t> elements_on_level(max_level + 1, 0);
  uint64_t running = 0;
  for (int l = max_level; l >= 0; --l) {
    running += assigned_to_level[l];
    elements_on_level[l] = running;
  }

  fmt::print("Build time: {:.2f} seconds\n", duration.count());
  fmt::print("Upper-level distance computations (layer > 0): {}\n", upper);
  fmt::print("Total distance computations: {}\n", total);
  fmt::print("Upper-level fraction: {:.4f}\n", frac);
  fmt::print("Max level: {}\n", max_level);
  fmt::print("Elements on each level (present on that layer):\n");
  for (int l = 0; l <= max_level; ++l) {
    fmt::print("  level {}: {}\n", l, elements_on_level[l]);
  }

  nlohmann::json json;
  json["datacard"] = args.datacard;
  json["k"] = args.k;
  json["M"] = args.M;
  json["efc"] = args.efc;
  json["n_base"] = c.n_base;
  json["vector_dim"] = c.vector_dim;
  json["hardware"] = "CPU-single";
  json["build_time_sec"] = duration.count();
  json["upper_level_distance_computations"] = upper;
  json["total_distance_computations"] = total;
  json["upper_level_fraction"] = frac;
  json["max_level"] = max_level;
  json["elements_on_level"] = elements_on_level;
  json["assigned_max_level"] = assigned_to_level;

  time_t ts = time(nullptr);
  auto tm = localtime(&ts);
  std::string json_file = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
  fs::path json_path = fs::path(LOGS) / method / workload / build / json_file;
  fs::create_directories(json_path.parent_path());
  fmt::print("Saving to {}\n", json_path.string());
  std::ofstream ofs(json_path.string());
  ofs << json.dump(4);
  ofs.close();

  return 0;
}
