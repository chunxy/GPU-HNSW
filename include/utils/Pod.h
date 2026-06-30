#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <vector>

using std::string;
using std::vector;
namespace po = boost::program_options;

struct DataCard {
  std::string name;
  std::string base_vector_path, query_vector_path;
  std::string groundtruth_path;
  // std::string base_point_path;
  // std::string query_type;  // sphere, rectangle
  uint32_t vector_dim;
  uint32_t n_base, n_queries, n_groundtruth;
};

// const int DIM = 2;
// struct Point {
//   float values[DIM];

//   bool is_within_sphere(const Point &center, const float radius) const {
//     float sum_squares = 0;
//     for (int i = 0; i < DIM; i++) {
//       float diff = values[i] - center.values[i];
//       sum_squares += diff * diff;
//     }
//     return sum_squares <= radius * radius;
//   }

//   bool is_within_rectangle(const Point &min, const Point &max) const {
//     for (int i = 0; i < DIM; i++) {
//       if (values[i] < min.values[i] || values[i] > max.values[i]) {
//         return false;
//       }
//     }
//     return true;
//   }
// };

struct Args {
  std::string datacard;
  int k = 100;
  int M = 32;
  int efc = 100;
  vector<int> efs = {100};
  vector<int> nprobe = {100};
  int nthread = 1;
  int batchsz = 100;
  bool build = true;
  bool profile = false;

  Args(int argc, char **argv) {
    po::options_description configs;
    po::options_description required_configs("Required"), optional_configs("Optional");
    // dataset parameter
    required_configs.add_options()("datacard", po::value<decltype(datacard)>(&datacard)->required());
    // search parameters
    required_configs.add_options()("k", po::value<decltype(k)>(&k)->required());
    // index constrcution parameters
    optional_configs.add_options()("M", po::value<decltype(M)>(&M));
    optional_configs.add_options()("efc", po::value<decltype(efc)>(&efc));
    // index search parameters
    optional_configs.add_options()("efs", po::value<decltype(efs)>(&efs)->multitoken());
    // system parameters
    optional_configs.add_options()("nthread", po::value<decltype(nthread)>(&nthread));
    optional_configs.add_options()("batchsz", po::value<decltype(batchsz)>(&batchsz));
    optional_configs.add_options()("build", po::value<decltype(build)>(&build));
    optional_configs.add_options()("profile", po::bool_switch(&profile), "Profile build_graph_kernel phases");
    // Merge required and optional configs.
    configs.add(required_configs).add(optional_configs);
    // Parse arguments.
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, configs), vm);
    po::notify(vm);
  }
};

struct QueryMetric {
  // core statistics
  int ncomp_cg;
  int nround;
  int ncomp;
  int ncluster;
  int nnodes;
  int cum_node_layer;
  int searched_graph_size;

  std::vector<bool> is_ivf_ppsl;
  std::vector<bool> is_graph_ppsl;
  std::vector<float> cand_dist;
  long long latency;
  long long filter_latency;
  long long cg_latency;
  long long graph_latency;
  long long ivf_latency;
  long long twohop_latency;
  long long ihnsw_latency;
  long long comp_latency;
  int ncomp_graph;
  int nrecycled;

  QueryMetric(int nb)
      : is_ivf_ppsl(nb, false),
        is_graph_ppsl(nb, false),
        latency(0),
        filter_latency(0),
        cg_latency(0),
        graph_latency(0),
        ivf_latency(0),
        twohop_latency(0),
        ihnsw_latency(0),
        comp_latency(0),
        ncomp_cg(0),
        nround(0),
        ncomp(0),
        nnodes(0),
        cum_node_layer(0),
        searched_graph_size(0),
        ncomp_graph(0),
        ncluster(0),
        nrecycled(0) {}
};

struct BatchMetric {
  std::vector<QueryMetric> qmetrics;
  long long time;
  long long overhead;
  long long cluster_search_time;

  BatchMetric(int nq, int nb) : qmetrics(nq, QueryMetric(nb)), time(0), overhead(0), cluster_search_time(0) {}
};

struct Stat {
  // per-query results
  vector<float> rec_at_ks;
  vector<float> pre_at_ks;
  vector<long> tp_s, rz_s;
  vector<float> gt_min_s, gt_max_s, rz_min_s, rz_max_s;
  vector<long> ivf_ppsl_in_rz_s, ivf_ppsl_in_tp_s;
  vector<long> graph_ppsl_in_rz_s, graph_ppsl_in_tp_s;
  // per-query intermediates
  vector<long> ivf_ppsl_nums;
  vector<float> ivf_ppsl_qlty;
  vector<float> ivf_ppsl_rate;
  vector<long> graph_ppsl_nums;
  vector<float> graph_ppsl_qlty;
  vector<float> graph_ppsl_rate;
  vector<vector<float>> cand_dist;
  vector<float> perc_of_ivf_ppsl_in_tp;
  vector<float> perc_of_ivf_ppsl_in_rz;
  vector<float> linear_scan_rate;
  vector<long> num_computations;
  vector<long> num_computations_graph;
  vector<long> cg_num_computations;
  vector<long> num_rounds;
  vector<long> num_clusters;
  vector<long> num_recycled;
  vector<long long> latencies;
  vector<long long> ivf_latencies;  // IVF latency includes CG latency and btree latency
  vector<long long> cg_latencies;
  vector<long long> filter_latencies;
  vector<long long> graph_latencies;
  vector<long long> twohop_latencies;
  vector<long long> misc_latencies;
  vector<long long> comp_latencies;
  // per-batch stat
  vector<long long> batch_time;
  vector<long long> batch_overhead;
  vector<long long> batch_cluster_search_time;  // leave it as is

  Stat(int nq)
      : rec_at_ks(nq, 0),
        pre_at_ks(nq, 0),
        tp_s(nq, 0),
        rz_s(nq, 0),
        gt_min_s(nq, 0),
        gt_max_s(nq, 0),
        rz_min_s(nq, 0),
        rz_max_s(nq, 0),
        ivf_ppsl_in_rz_s(nq, 0),
        ivf_ppsl_in_tp_s(nq, 0),
        graph_ppsl_in_rz_s(nq, 0),
        graph_ppsl_in_tp_s(nq, 0),
        ivf_ppsl_nums(nq, 0),
        ivf_ppsl_qlty(nq, 0),
        ivf_ppsl_rate(nq, 0),
        graph_ppsl_nums(nq, 0),
        graph_ppsl_qlty(nq, 0),
        graph_ppsl_rate(nq, 0),
        cand_dist(nq),
        perc_of_ivf_ppsl_in_tp(nq, 0),
        perc_of_ivf_ppsl_in_rz(nq, 0),
        linear_scan_rate(nq, 0),
        num_computations(nq, 0),
        num_computations_graph(nq, 0),
        cg_num_computations(nq, 0),
        num_rounds(nq, 0),
        num_clusters(nq, 0),
        num_recycled(nq, 0),
        latencies(nq, 0),
        cg_latencies(nq, 0),
        filter_latencies(nq, 0),
        graph_latencies(nq, 0),
        ivf_latencies(nq, 0),
        twohop_latencies(nq, 0),
        misc_latencies(nq, 0),
        comp_latencies(nq, 0) {}
};