#include <fmt/chrono.h>
#include <fmt/format.h>
#include <algorithm>
#include <boost/program_options.hpp>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "config.h"
#include "hnswalg-lite.h"
#include "json.hpp"
#include "utils/Pod.h"
#include "utils/funcs.h"
#include "utils/reader.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {
struct BenchLoadArgs {
  std::string datacard;
  int k = 10;
  int M = 32;
  int efc = 100;
  vector<int> efs = {100};
  std::string hardware;
};

std::string trim_lower(std::string s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return s;
}

void resolve_hardware(const std::string &raw, std::string &method, std::string &suffix, std::string &log_hardware) {
  const std::string hw = trim_lower(raw);
  if (hw == "gpu") {
    method = "GPU";
    suffix = "gpu";
    log_hardware = "gpu";
    return;
  }
  if (hw == "cpu") {
    method = "CPU";
    suffix = "cpu";
    log_hardware = "cpu";
    return;
  }
  if (hw == "cpu-single" || hw == "single-thread cpu" || hw == "single thread cpu" || hw == "single-cpu" ||
      hw == "cpu single") {
    method = "CPU-single";
    suffix = "cpu-single";
    log_hardware = "cpu-single";
    return;
  }
  throw std::runtime_error(fmt::format("Unsupported --hardware '{}'. Use one of: GPU, CPU, single-thread CPU", raw));
}

BenchLoadArgs parse_args(int argc, char **argv) {
  BenchLoadArgs args;
  po::options_description desc("bench-load options");
  desc.add_options()("help,h", "Show help message");
  desc.add_options()("datacard", po::value<std::string>(&args.datacard)->required(), "Dataset card name");
  desc.add_options()("k", po::value<int>(&args.k)->required(), "Top-k for evaluation");
  desc.add_options()(
      "hardware", po::value<std::string>(&args.hardware)->required(), "Build hardware: GPU|CPU|single-thread CPU");
  desc.add_options()("M", po::value<int>(&args.M)->default_value(args.M), "Graph M");
  desc.add_options()("efc", po::value<int>(&args.efc)->default_value(args.efc), "Construction ef");
  desc.add_options()("efs", po::value<std::vector<int>>(&args.efs)->required()->multitoken(), "Search ef");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  if (vm.count("help") > 0) {
    std::cout << desc << '\n';
    std::exit(0);
  }
  po::notify(vm);
  return args;
}
}  // namespace

int main(int argc, char **argv) {
  BenchLoadArgs args = parse_args(argc, argv);
  extern std::map<std::string, DataCard> name_to_card;
  if (name_to_card.find(args.datacard) == name_to_card.end()) {
    throw std::runtime_error(fmt::format("Unknown datacard '{}'", args.datacard));
  }
  DataCard c = name_to_card[args.datacard];
  if (args.k > static_cast<int>(c.n_groundtruth)) {
    throw std::runtime_error(
        fmt::format("k({}) is greater than the number of groundtruth({})", args.k, c.n_groundtruth));
  }

  std::string method, suffix, log_hardware;
  resolve_hardware(args.hardware, method, suffix, log_hardware);

  const std::string index_name = fmt::format(fmt::runtime(CHECKPOINT_TMPL), args.datacard, args.M, args.efc, suffix);
  const fs::path index_path = fs::path(CKPS) / method / index_name;
  if (!fs::exists(index_path)) {
    throw std::runtime_error(fmt::format("Saved index not found at '{}'", index_path.string()));
  }

  float *xb = nullptr;
  float *xq = nullptr;
  load_base_and_query(c, xb, xq);

  IVecItrReader gt_reader(c.groundtruth_path);
  std::vector<std::vector<uint32_t>> gt(c.n_queries);
  size_t gt_idx = 0;
  while (!gt_reader.HasEnded() && gt_idx < c.n_queries) {
    gt[gt_idx] = gt_reader.Next();
    gt_idx++;
  }
  if (gt_idx != c.n_queries) {
    throw std::runtime_error(fmt::format("Groundtruth query count mismatch. expected={}, got={}", c.n_queries, gt_idx));
  }
  for (size_t i = 0; i < c.n_queries; ++i) {
    if (gt[i].size() < static_cast<size_t>(args.k)) {
      throw std::runtime_error(
          fmt::format("Groundtruth top-k too short at query {}. expected >= {}, got {}", i, args.k, gt[i].size()));
    }
  }

  hnswlib::L2Space space(c.vector_dim);
  hnswlib::HierarchicalNswLite<float> index(&space);
  index.loadIndex(index_path.string(), &space, c.n_base);
  index.checkIntegrity();
  for (int efs : args.efs) {
    index.setEf(std::max(args.k, efs));
    size_t hit_count = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (size_t qi = 0; qi < c.n_queries; ++qi) {
      auto result = index.searchKnn(xq + qi * c.vector_dim, args.k);
      std::unordered_set<uint32_t> gt_set;
      gt_set.reserve(args.k);
      for (int j = 0; j < args.k; ++j) {
        gt_set.insert(gt[qi][j]);
      }
      while (!result.empty()) {
        const uint32_t id = static_cast<uint32_t>(result.top().second);
        result.pop();
        if (gt_set.find(id) != gt_set.end()) {
          hit_count++;
        }
      }
    }
    auto t1 = std::chrono::steady_clock::now();

    const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    const double recall = static_cast<double>(hit_count) / static_cast<double>(c.n_queries * args.k);
    const double qps = static_cast<double>(c.n_queries) / elapsed_s;

    nlohmann::json json;
    json["recall"] = recall;
    json["qps"] = qps;
    json["dataset"] = args.datacard;
    json["hardware"] = log_hardware;
    json["build"] = {{"M", args.M}, {"efc", args.efc}};
    json["search"] = {{"k", args.k}, {"efs", efs}};

    time_t ts = time(nullptr);
    auto tm = localtime(&ts);
    const std::string json_file = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
    const fs::path json_path = fs::path(LOGS) / log_hardware / args.datacard /
                               fmt::format("M_{}_efc_{}", args.M, args.efc) / fmt::format("k_{}_efs_{}", args.k, efs) /
                               json_file;
    fs::create_directories(json_path.parent_path());
    std::ofstream ofs(json_path.string());
    ofs << json.dump(4);
    ofs.close();

    fmt::print("Recall: {:.6f}\n", recall);
    fmt::print("QPS: {:.2f}\n", qps);
    fmt::print("Saved to {}\n", json_path.string());
  }

  return 0;
}
