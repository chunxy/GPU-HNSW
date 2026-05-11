#include <fmt/chrono.h>
#include <fmt/format.h>
#include <omp.h>
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
    fmt::print("k({}) is greater than the number of groundtruth({}). Exit.\n", args.k, c.n_groundtruth);
    exit(-1);
  }

  // Load data.
  float *xb, *xq;
  load_base_and_query(c, xb, xq);

  string method = "CPU-single";
  string suffix = method;
  std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
  string index_name = fmt::format(fmt::runtime(CHECKPOINT_TMPL), args.datacard, args.M, args.efc, suffix);
  string index_path = fs::path(CKPS) / method / index_name;
  fs::create_directories(fs::path(index_path).parent_path());

  nlohmann::json json;

  string workload = fmt::format(fmt::runtime(WORKLOAD_TMPL), args.datacard, args.k);
  string build = fmt::format("M_{}_efc_{}", args.M, args.efc);
  // string search = fmt::format("efs_{}", efs);

  hnswlib::L2Space space(c.vector_dim);
  hnswlib::HierarchicalNswLite<float> index(&space, c.n_base, args.M, args.efc);
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  // set the number of threads to the number of cores in the system
  // omp_set_num_threads(omp_get_num_procs() / 2);
  omp_set_num_threads(1);
#pragma omp parallel for
  for (size_t i = 0; i < c.n_base; ++i) {
    index.addPoint(xb + i * c.vector_dim, i);
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = end - start;
  index.saveIndex(index_path);

  fmt::print("Build time: {:.2f} seconds\n", duration.count());

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