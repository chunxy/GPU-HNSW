#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <omp.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include "utils/Pod.h"
#include "utils/funcs.h"

using std::pair;
using std::priority_queue;
using std::vector;

namespace po = boost::program_options;
using namespace std::chrono;

void compute_groundtruth(
    const float *xb,
    const int nb,
    const float *xq,
    const int nq,
    const size_t d,
    const int k,
    vector<vector<pair<float, uint32_t>>> &gt) {
  vector<priority_queue<pair<float, uint32_t>>> pq_topks(nq);
  gt.resize(nq);

  hnswlib::L2Space space(d);
  auto compute_start = high_resolution_clock::now();
  omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nq; i++) {
    const float *query = xq + i * d;

    for (int j = 0; j < nb; j++) {
      auto dist = space.get_dist_func()(query, xb + j * d, &d);
      pq_topks[i].emplace(dist, j);
      while (pq_topks[i].size() > k) pq_topks[i].pop();
    }
    gt[i].resize(pq_topks[i].size());
    int sz = pq_topks[i].size();
    while (pq_topks[i].size() != 0) {
      gt[i][--sz] = pq_topks[i].top();
      pq_topks[i].pop();
    }
  }
  auto compute_end = high_resolution_clock::now();
  fmt::print("Computation took {} microseconds\n", duration_cast<microseconds>(compute_end - compute_start).count());
}

int main(int argc, char **argv) {
  std::string dataname;

  po::options_description configs;
  configs.add_options()("datacard", po::value<decltype(dataname)>(&dataname)->required());
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, configs), vm);
  po::notify(vm);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[dataname];
  int nb = c.n_base, nq = c.n_queries, d = c.vector_dim;
  uint32_t k = c.n_groundtruth;

  float *xb, *xq;
  load_base_and_query(c, xb, xq);

  vector<vector<pair<float, uint32_t>>> gt(nq);
  compute_groundtruth(xb, nb, xq, nq, d, k, gt);

  std::ofstream ofs(c.groundtruth_path, std::ios_base::binary & std::ios_base::out);
  for (int i = 0; i < nq; i++) {
    uint32_t size = gt[i].size();
    ofs.write((char *)&size, sizeof(size));
    for (int j = 0; j < gt[i].size(); j++) {
      ofs.write((char *)&gt[i][j].second, sizeof(uint32_t));
    }
  }
  ofs.close();

  return 0;
}