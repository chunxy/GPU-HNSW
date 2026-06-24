#include <fmt/chrono.h>
#include <fmt/format.h>
#include <boost/program_options.hpp>
#include <filesystem>
#include "config.h"
#include "hnswalg-lite.h"
#include "json.hpp"
#include "utils/Pod.h"
#include "utils/funcs.h"

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  Args args(argc, argv);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[args.datacard];

  if (args.k > c.n_groundtruth) {
    fmt::print("k is greater than the number of groundtruth. Exit.\n");
    exit(-1);
  }

  // Load data.
  float *xb, *xq;
  load_base_and_query(c, xb, xq);

  string method = "GPU";
  string suffix = method;
  std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
  string index_name = fmt::format(fmt::runtime(CHECKPOINT_TMPL), args.datacard, args.M, args.efc, suffix);
  string index_path = fs::path(CKPS) / method / index_name;
  fs::create_directories(fs::path(index_path).parent_path());

  nlohmann::json json;

  string workload = fmt::format(fmt::runtime(WORKLOAD_TMPL), args.datacard, args.k);
  string build = fmt::format("M_{}_efc_{}", args.M, args.efc);
  // string search = fmt::format("efs_{}", efs);

  hnswlib::HierarchicalNswLite<float> index(c.vector_dim, c.n_base, xb, args.M, args.efc);
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  index.build_graph_gpu();
  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = end - start;
  fmt::print("Build time: {:.2f} seconds\n", duration.count());

  index.copy_from_gpu();
  index.saveIndex(index_path);

  return 0;
}