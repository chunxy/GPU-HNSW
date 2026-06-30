#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <stdlib.h>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "hnswlib/hnswlib.h"
#include "hnswlib/visited_list_pool.h"
#include "kernel.cuh"

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

static uint32_t checked_u32(size_t value, const char *name) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error(std::string(name) + " exceeds the 32-bit GPU limit");
  }
  return static_cast<uint32_t>(value);
}

static uint32_t checked_label_u32(labeltype value, const char *name) {
  if (value > static_cast<labeltype>(std::numeric_limits<uint32_t>::max())) {
    throw std::overflow_error(std::string(name) + " exceeds the 32-bit GPU limit");
  }
  return static_cast<uint32_t>(value);
}

static void cudaCheck(cudaError_t code, const char *expr, const char *file, int line) {
  if (code == cudaSuccess) {
    return;
  }
  std::ostringstream oss;
  oss << "CUDA failure for " << expr << " at " << file << ":" << line << ": " << cudaGetErrorString(code);
  fmt::print("{}\n", oss.str());
  exit(-1);
  // throw std::runtime_error(oss.str());
}

static void cuda_free_buffer(void *ptr) {
  if (ptr != nullptr) {
    (void)cudaFree(ptr);
  }
}

template <typename dist_t>
class HierarchicalNswLite {
 public:
  static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;

  using GpuFreeFn = void (*)(void *);

  size_t max_elements_{0};
  mutable std::atomic<size_t> cur_element_count{0};
  size_t size_data_per_element_{0};
  size_t size_links_per_element_{0};
  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_construction_{0};
  size_t ef_{0};

  double mult_{0.0}, revSize_{0.0};
  int maxlevel_{0};

  std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

  mutable std::vector<std::mutex> label_op_locks_;
  std::mutex global;
  std::vector<std::mutex> link_list_locks_;

  tableint enterpoint_node_{0};

  size_t size_links_level0_{0};
  size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

  char *data_level0_memory_{nullptr};
  char **linkLists_{nullptr};
  std::vector<int> element_levels_;

  size_t data_size_{0};

  DISTFUNC<dist_t> fstdistfunc_;
  void *dist_func_param_{nullptr};

  mutable std::mutex label_lookup_lock;
  std::unordered_map<labeltype, tableint> label_lookup_;

  std::default_random_engine level_generator_;

  mutable std::atomic<long> metric_distance_computations{0};
  mutable std::atomic<long> metric_hops{0};

  GpuGraphState *device_gpu_graph_state_{nullptr};

  GpuGraphState host_gpu_graph_state_{};
  float *d_vector_data_{nullptr};

  GpuFreeFn gpu_free_fn_{nullptr};
  bool gpu_graph_ready_{false};

  HierarchicalNswLite(SpaceInterface<dist_t> *s) {}

  HierarchicalNswLite(
      SpaceInterface<dist_t> *s,
      const std::string &location,
      bool /* nmslib */ = false,
      size_t max_elements = 0) {
    loadIndex(location, s, max_elements);
  }

  HierarchicalNswLite(
      SpaceInterface<dist_t> *s,
      size_t max_elements,
      size_t M = 16,
      size_t ef_construction = 200,
      size_t random_seed = 100)
      : label_op_locks_(MAX_LABEL_OPERATION_LOCKS), link_list_locks_(max_elements), element_levels_(max_elements) {
    max_elements_ = max_elements;
    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    if (M <= 10000) {
      M_ = M;
    } else {
      HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
      HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
      M_ = 10000;
    }
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    level_generator_.seed(random_seed);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr) throw std::runtime_error("Not enough memory");

    cur_element_count = 0;

    visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
    if (linkLists_ == nullptr)
      throw std::runtime_error("Not enough memory: HierarchicalNSWStatic failed to allocate linklists");
    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * M_);
    revSize_ = 1.0 / mult_;
  }

  // GPU-version constructor
  HierarchicalNswLite(
      size_t data_dim,
      size_t max_elements,
      float *vector_data,
      size_t M = 16,
      size_t ef_construction = 200,
      size_t random_seed = 100)
      : label_op_locks_(MAX_LABEL_OPERATION_LOCKS), link_list_locks_(max_elements), element_levels_(max_elements) {
    max_elements_ = max_elements;
    data_size_ = sizeof(float) * data_dim;
    // fstdistfunc_ = s->get_dist_func();
    // dist_func_param_ = s->get_dist_func_param();
    if (M <= 10000) {
      M_ = M;
    } else {
      HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
      HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
      M_ = 10000;
    }
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    level_generator_.seed(random_seed);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr) throw std::runtime_error("Not enough memory");

    cur_element_count = 0;

    visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
    if (linkLists_ == nullptr)
      throw std::runtime_error("Not enough memory: HierarchicalNSWStatic failed to allocate linklists");
    memset(linkLists_, 0, sizeof(void *) * max_elements_);
    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * M_);
    revSize_ = 1.0 / mult_;

    d_vector_data_ = vector_data;

    move_to_gpu();
    // build_graph_gpu();
  }

  ~HierarchicalNswLite() { clear(); }

  void clear() {
    release_gpu_state();
    free(data_level0_memory_);
    data_level0_memory_ = nullptr;
    for (tableint i = 0; i < cur_element_count; i++) {
      if (element_levels_[i] > 0) free(linkLists_[i]);
    }
    free(linkLists_);
    linkLists_ = nullptr;
    cur_element_count = 0;
    visited_list_pool_.reset(nullptr);
  }

  struct CompareByFirst {
    constexpr bool operator()(std::pair<dist_t, tableint> const &a, std::pair<dist_t, tableint> const &b)
        const noexcept {
      return a.first < b.first;
    }
  };

  void release_gpu_state() {
    if (gpu_free_fn_ != nullptr) {
      gpu_free_fn_(host_gpu_graph_state_.precomputed_new_new_dist);
      gpu_free_fn_(host_gpu_graph_state_.precomputed_new_new_rank);
      gpu_free_fn_(host_gpu_graph_state_.news_dist);
      gpu_free_fn_(host_gpu_graph_state_.news_rank);
      gpu_free_fn_(host_gpu_graph_state_.frozen_link_counts);
      gpu_free_fn_(host_gpu_graph_state_.changed_old_links);
      gpu_free_fn_(host_gpu_graph_state_.changed_old_link_counts);
      gpu_free_fn_(device_gpu_graph_state_);
      // gpu_free_fn_(host_gpu_graph_state_.half_vector_data_);
    }

    device_gpu_graph_state_ = nullptr;
    host_gpu_graph_state_ = GpuGraphState{};
    gpu_free_fn_ = nullptr;
    gpu_graph_ready_ = false;
  }

  const GpuGraphState &gpu_graph_state() const { return host_gpu_graph_state_; }

  const GpuGraphState *device_gpu_graph_state() const {
    return static_cast<const GpuGraphState *>(device_gpu_graph_state_);
  }

  void setEf(size_t ef) { ef_ = ef; }

  inline std::mutex &getLabelOpMutex(labeltype label) const {
    size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
    return label_op_locks_[lock_id];
  }

  inline labeltype getExternalLabel(tableint internal_id) const {
    labeltype return_label;
    memcpy(
        &return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
    return return_label;
  }

  inline void setExternalLabel(tableint internal_id, labeltype label) const {
    memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
  }

  inline labeltype *getExternalLabeLp(tableint internal_id) const {
    return (labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
  }

  inline char *getDataByInternalId(tableint internal_id) const {
    return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
  }

  int getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
  }

  size_t getMaxElements() { return max_elements_; }

  size_t getCurrentElementCount() { return cur_element_count; }

  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
  searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        candidateSet;

    dist_t lowerBound;
    dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
    top_candidates.emplace(dist, ep_id);
    lowerBound = dist;
    candidateSet.emplace(-dist, ep_id);
    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
      std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
      if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
        break;
      }
      candidateSet.pop();

      tableint curNodeNum = curr_el_pair.second;

      std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

      int *data;
      if (layer == 0) {
        data = (int *)get_linklist0(curNodeNum);
      } else {
        data = (int *)get_linklist(curNodeNum, layer);
      }
      size_t size = getListCount((linklistsizeint *)data);
      tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(datal + j);
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
        if (visited_array[candidate_id] == visited_array_tag) {
          continue;
        }
        visited_array[candidate_id] = visited_array_tag;
        char *currObj1 = (getDataByInternalId(candidate_id));

        dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
        if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
          candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
          _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

          top_candidates.emplace(dist1, candidate_id);

          if (top_candidates.size() > ef_construction_) top_candidates.pop();

          if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
        }
      }
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
  }

  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
  searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef, BaseFilterFunctor *isIdAllowed = nullptr) const {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        candidate_set;

    char *ep_data = getDataByInternalId(ep_id);
    dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
    dist_t lowerBound = std::numeric_limits<dist_t>::max();
    if ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))) {
      lowerBound = dist;
      top_candidates.emplace(dist, ep_id);
    }
    candidate_set.emplace(-dist, ep_id);

    visited_array[ep_id] = visited_array_tag;

    while (!candidate_set.empty()) {
      std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
      dist_t candidate_dist = -current_node_pair.first;
      if (candidate_dist > lowerBound && top_candidates.size() == ef) {
        break;
      }
      candidate_set.pop();

      tableint current_node_id = current_node_pair.second;
      int *data = (int *)get_linklist0(current_node_id);
      size_t size = getListCount((linklistsizeint *)data);

#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
      _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

      for (size_t j = 1; j <= size; j++) {
        int candidate_id = *(data + j);
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
        _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
#endif
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;

          char *currObj1 = (getDataByInternalId(candidate_id));
          dist_t candidate_distance = fstdistfunc_(data_point, currObj1, dist_func_param_);

          if (top_candidates.size() < ef || lowerBound > candidate_distance) {
            candidate_set.emplace(-candidate_distance, candidate_id);
#ifdef USE_SSE
            _mm_prefetch(
                data_level0_memory_ + candidate_set.top().second * size_data_per_element_ + offsetLevel0_, _MM_HINT_T0);
#endif

            if ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))) {
              top_candidates.emplace(candidate_distance, candidate_id);
            }

            while (top_candidates.size() > ef) {
              top_candidates.pop();
            }

            if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
          }
        }
      }
    }

    visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
  }

  void getNeighborsByHeuristic2(
      std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
          &top_candidates,
      const size_t M) {
    if (top_candidates.size() < M) {
      return;
    }

    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0) {
      queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
      top_candidates.pop();
    }

    while (queue_closest.size()) {
      if (return_list.size() >= M) break;
      std::pair<dist_t, tableint> curent_pair = queue_closest.top();
      dist_t dist_to_query = -curent_pair.first;
      queue_closest.pop();
      bool good = true;

      for (std::pair<dist_t, tableint> second_pair : return_list) {
        dist_t curdist = fstdistfunc_(
            getDataByInternalId(second_pair.second), getDataByInternalId(curent_pair.second), dist_func_param_);
        if (curdist < dist_to_query) {
          good = false;
          break;
        }
      }
      if (good) {
        return_list.push_back(curent_pair);
      }
    }

    for (std::pair<dist_t, tableint> curent_pair : return_list) {
      top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
  }

  linklistsizeint *get_linklist0(tableint internal_id) const {
    return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
  }

  linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory) const {
    return (linklistsizeint *)(data_level0_memory + internal_id * size_data_per_element_ + offsetLevel0_);
  }

  linklistsizeint *get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint *)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
  }

  linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
    return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
  }

  tableint mutuallyConnectNewElement(
      const void *data_point,
      tableint cur_c,
      std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
          &top_candidates,
      int level) {
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, M_);
    if (top_candidates.size() > M_)
      throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(M_);
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.back();

    linklistsizeint *ll_cur;
    if (level == 0)
      ll_cur = get_linklist0(cur_c);
    else
      ll_cur = get_linklist(cur_c, level);

    if (*ll_cur) {
      throw std::runtime_error("The newly inserted element should have blank link list");
    }

    setListCount(ll_cur, selectedNeighbors.size());
    tableint *data = (tableint *)(ll_cur + 1);
    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      if (data[idx]) throw std::runtime_error("Possible memory corruption");
      if (level > element_levels_[selectedNeighbors[idx]])
        throw std::runtime_error("Trying to make a link on a non-existent level");

      data[idx] = selectedNeighbors[idx];
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

      linklistsizeint *ll_other;
      if (level == 0)
        ll_other = get_linklist0(selectedNeighbors[idx]);
      else
        ll_other = get_linklist(selectedNeighbors[idx], level);

      size_t sz_link_list_other = getListCount(ll_other);

      if (sz_link_list_other > Mcurmax) throw std::runtime_error("Bad value of sz_link_list_other");
      if (selectedNeighbors[idx] == cur_c) throw std::runtime_error("Trying to connect an element to itself");
      if (level > element_levels_[selectedNeighbors[idx]])
        throw std::runtime_error("Trying to make a link on a non-existent level");

      tableint *other_data = (tableint *)(ll_other + 1);

      if (sz_link_list_other < Mcurmax) {
        other_data[sz_link_list_other] = cur_c;
        setListCount(ll_other, sz_link_list_other + 1);
      } else {
        dist_t d_max =
            fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            candidates;
        candidates.emplace(d_max, cur_c);

        for (size_t j = 0; j < sz_link_list_other; j++) {
          candidates.emplace(
              fstdistfunc_(
                  getDataByInternalId(other_data[j]), getDataByInternalId(selectedNeighbors[idx]), dist_func_param_),
              other_data[j]);
        }

        getNeighborsByHeuristic2(candidates, Mcurmax);

        int indx = 0;
        while (candidates.size() > 0) {
          other_data[indx] = candidates.top().second;
          candidates.pop();
          indx++;
        }

        setListCount(ll_other, indx);
      }
    }

    return next_closest_entry_point;
  }

  void resizeIndex(size_t new_max_elements) {
    if (new_max_elements < cur_element_count)
      throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

    visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));
    element_levels_.resize(new_max_elements);
    std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

    char *data_level0_memory_new = (char *)realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
    if (data_level0_memory_new == nullptr)
      throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
    data_level0_memory_ = data_level0_memory_new;

    char **linkLists_new = (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
    if (linkLists_new == nullptr)
      throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
    linkLists_ = linkLists_new;

    max_elements_ = new_max_elements;
  }

  size_t indexFileSize() const {
    size_t size = 0;
    size += sizeof(offsetLevel0_);
    size += sizeof(max_elements_);
    size += sizeof(cur_element_count);
    size += sizeof(size_data_per_element_);
    size += sizeof(label_offset_);
    size += sizeof(offsetData_);
    size += sizeof(maxlevel_);
    size += sizeof(enterpoint_node_);
    size += sizeof(maxM_);
    size += sizeof(maxM0_);
    size += sizeof(M_);
    size += sizeof(mult_);
    size += sizeof(ef_construction_);

    size += cur_element_count * size_data_per_element_;

    for (size_t i = 0; i < cur_element_count; i++) {
      unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
      size += sizeof(linkListSize);
      size += linkListSize;
    }
    return size;
  }

  void saveIndex(const std::string &location) {
    std::ofstream output(location, std::ios::binary);

    writeBinaryPOD(output, offsetLevel0_);
    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, label_offset_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, maxM_);
    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, ef_construction_);

    output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

    for (size_t i = 0; i < cur_element_count; i++) {
      unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
      writeBinaryPOD(output, linkListSize);
      if (linkListSize) output.write(linkLists_[i], linkListSize);
    }
    output.close();
  }

  void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
    std::ifstream input(location, std::ios::binary);

    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    clear();
    input.seekg(0, input.end);
    std::streampos total_filesize = input.tellg();
    input.seekg(0, input.beg);

    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count);

    size_t max_elements = max_elements_i;
    if (max_elements < cur_element_count) max_elements = max_elements_;
    max_elements_ = max_elements;
    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);
    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);

    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();

    auto pos = input.tellg();

    input.seekg(cur_element_count * size_data_per_element_, input.cur);
    for (size_t i = 0; i < cur_element_count; i++) {
      if (input.tellg() < 0 || input.tellg() >= total_filesize) {
        throw std::runtime_error("Index seems to be corrupted or unsupported");
      }

      unsigned int linkListSize;
      readBinaryPOD(input, linkListSize);
      if (linkListSize != 0) {
        input.seekg(linkListSize, input.cur);
      }
    }

    if (input.tellg() != total_filesize) throw std::runtime_error("Index seems to be corrupted or unsupported");

    input.clear();
    input.seekg(pos, input.beg);

    data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    std::vector<std::mutex>(max_elements).swap(link_list_locks_);
    std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

    visited_list_pool_.reset(new VisitedListPool(1, max_elements));

    linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
    if (linkLists_ == nullptr) throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
    element_levels_ = std::vector<int>(max_elements);
    revSize_ = 1.0 / mult_;
    ef_ = 10;
    for (size_t i = 0; i < cur_element_count; i++) {
      label_lookup_[getExternalLabel(i)] = i;
      unsigned int linkListSize;
      readBinaryPOD(input, linkListSize);
      if (linkListSize == 0) {
        element_levels_[i] = 0;
        linkLists_[i] = nullptr;
      } else {
        element_levels_[i] = linkListSize / size_links_per_element_;
        linkLists_[i] = (char *)malloc(linkListSize);
        if (linkLists_[i] == nullptr)
          throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
        input.read(linkLists_[i], linkListSize);
      }
    }

    input.close();
  }

  template <typename data_t>
  std::vector<data_t> getDataByLabel(labeltype label) const {
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock<std::mutex> lock_table(label_lookup_lock);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
      throw std::runtime_error("Label not found");
    }
    tableint internalId = search->second;
    lock_table.unlock();

    char *data_ptrv = getDataByInternalId(internalId);
    size_t dim = *((size_t *)dist_func_param_);
    std::vector<data_t> data;
    data_t *data_ptr = (data_t *)data_ptrv;
    for (size_t i = 0; i < dim; i++) {
      data.push_back(*data_ptr);
      data_ptr += 1;
    }
    return data;
  }

  unsigned short int getListCount(linklistsizeint *ptr) const { return *((unsigned short int *)ptr); }

  void setListCount(linklistsizeint *ptr, unsigned short int size) const {
    *((unsigned short int *)ptr) = *((unsigned short int *)&size);
  }

  void addPoint(const void *data_point, labeltype label) {
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
    addPoint(data_point, label, -1);
  }

  tableint addPoint(const void *data_point, labeltype label, int level) {
    tableint cur_c = 0;
    {
      std::unique_lock<std::mutex> lock_table(label_lookup_lock);
      auto search = label_lookup_.find(label);
      if (search != label_lookup_.end()) {
        throw std::runtime_error("The element with the same label already exists");
      }

      if (cur_element_count >= max_elements_) {
        throw std::runtime_error("The number of elements exceeds the specified limit");
      }

      cur_c = cur_element_count;
      cur_element_count++;
      label_lookup_[label] = cur_c;
    }

    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
    int curlevel = getRandomLevel(mult_);
    if (level > 0) curlevel = level;

    element_levels_[cur_c] = curlevel;

    std::unique_lock<std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy) templock.unlock();
    tableint currObj = enterpoint_node_;

    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);

    if (curlevel) {
      linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * curlevel + 1);
      if (linkLists_[cur_c] == nullptr)
        throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
      memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
    }

    if ((signed)currObj != -1) {
      if (curlevel < maxlevelcopy) {
        dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
        for (int layer = maxlevelcopy; layer > curlevel; layer--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;
            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
            data = get_linklist(currObj, layer);
            int size = getListCount(data);

            tableint *datal = (tableint *)(data + 1);
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];
              if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");
              dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
              if (d < curdist) {
                curdist = d;
                currObj = cand;
                changed = true;
              }
            }
          }
        }
      }

      for (int layer = std::min(curlevel, maxlevelcopy); layer >= 0; layer--) {
        if (layer > maxlevelcopy || layer < 0) throw std::runtime_error("Level error");

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            top_candidates = searchBaseLayer(currObj, data_point, layer);
        currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, layer);
      }
    } else {
      enterpoint_node_ = 0;
      maxlevel_ = curlevel;
    }

    if (curlevel > maxlevelcopy) {
      enterpoint_node_ = cur_c;
      maxlevel_ = curlevel;
    }
    return cur_c;
  }

  std::priority_queue<std::pair<dist_t, labeltype>>
  searchKnn(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0) {
      return result;
    }

    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        unsigned int *data;

        data = (unsigned int *)get_linklist(currObj, level);
        int size = getListCount(data);
        metric_hops++;
        metric_distance_computations += size;

        tableint *datal = (tableint *)(data + 1);
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");
          dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        top_candidates = searchBaseLayerST(currObj, query_data, std::max(ef_, k), isIdAllowed);

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
      std::pair<dist_t, tableint> rez = top_candidates.top();
      result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
      top_candidates.pop();
    }
    return result;
  }

  std::vector<std::pair<dist_t, labeltype>>
  searchKnnCloserFirst(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
    std::vector<std::pair<dist_t, labeltype>> result;
    auto ret = searchKnn(query_data, k, isIdAllowed);
    size_t sz = ret.size();
    result.resize(sz);
    while (!ret.empty()) {
      result[--sz] = ret.top();
      ret.pop();
    }
    return result;
  }

  void checkIntegrity() {
    int connections_checked = 0;
    std::vector<int> inbound_connections_num(cur_element_count, 0);
    for (int i = 0; i < cur_element_count; i++) {
      for (int l = 0; l <= element_levels_[i]; l++) {
        linklistsizeint *ll_cur = get_linklist_at_level(i, l);
        int size = getListCount(ll_cur);
        tableint *data = (tableint *)(ll_cur + 1);
        std::unordered_set<tableint> s;
        for (int j = 0; j < size; j++) {
          assert(data[j] < cur_element_count);
          assert(data[j] != i);
          inbound_connections_num[data[j]]++;
          s.insert(data[j]);
          connections_checked++;
        }
        assert(s.size() == size);
      }
    }
    if (cur_element_count > 1) {
      int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
      for (int i = 0; i < cur_element_count; i++) {
        assert(inbound_connections_num[i] > 0);
        min1 = std::min(inbound_connections_num[i], min1);
        max1 = std::max(inbound_connections_num[i], max1);
      }
      std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
    }
    std::cout << "integrity ok, checked " << connections_checked << " connections\n";
  }

  void move_to_gpu() {
    release_gpu_state();

    const size_t cur_count = cur_element_count.load();
    const uint32_t gpu_cur_count = checked_u32(cur_count, "cur_element_count");
    const uint32_t gpu_max_elements = checked_u32(max_elements_, "max_elements");
    const uint32_t gpu_vector_dim = checked_u32(data_size_ / sizeof(float), "vector_dim");
    const uint32_t gpu_size_links_level0 =
        checked_u32(sizeof(linklistsizeint) + sizeof(tableint) * (maxM0_ + BATCHSZ_PER_NEW) * 2, "size_links_level0");
    const uint32_t gpu_size_links_per_element =
        checked_u32(sizeof(linklistsizeint) + sizeof(tableint) * (M_ + BATCHSZ_PER_NEW) * 2, "size_links_per_element");
    const uint32_t gpu_data_size = checked_u32(data_size_, "data_size");
    const size_t old_new_distances_count = (BATCHSZ_PER_NEW) * (LEVEL_SZ_THRES + BATCHSZ_PER_NEW);
    const size_t num_new_new_batches = (gpu_max_elements + BATCHSZ_PER_NEW - 1) / BATCHSZ_PER_NEW;
    const size_t precomputed_new_new_count = num_new_new_batches * BATCHSZ_PER_NEW * BATCHSZ_PER_NEW;

    auto malloc_and_copy = [&](void **dst, const void *src, size_t bytes) {
      if (bytes == 0) {
        *dst = nullptr;
        return;
      }
      cudaCheck(cudaMalloc(dst, bytes), "cudaMalloc", __FILE__, __LINE__);
      cudaCheck(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    };

    try {
      malloc_and_copy(
          (void **)&host_gpu_graph_state_.vector_data,
          static_cast<void *>(d_vector_data_),
          gpu_max_elements * gpu_vector_dim * sizeof(float));
      // host_gpu_graph_state_.level0_links = static_cast<char *>(d_level0_links_);
      // host_gpu_graph_state_.vector_data = static_cast<const float *>(d_vector_data_);

      host_gpu_graph_state_.max_elements = gpu_max_elements;
      host_gpu_graph_state_.cur_element_count = gpu_cur_count;
      host_gpu_graph_state_.size_data_per_element = checked_u32(size_data_per_element_, "size_data_per_element");
      host_gpu_graph_state_.ef_construction = checked_u32(ef_construction_, "ef_construction");
      host_gpu_graph_state_.ef = checked_u32(ef_, "ef");
      host_gpu_graph_state_.enterpoint_node =
          cur_count == 0 ? kInvalidGpuOffset : static_cast<uint32_t>(enterpoint_node_);
      host_gpu_graph_state_.M = checked_u32(M_, "M");
      host_gpu_graph_state_.maxM0 = checked_u32(maxM0_, "maxM0");
      host_gpu_graph_state_.size_links_level0 = gpu_size_links_level0;
      host_gpu_graph_state_.size_links_per_element = gpu_size_links_per_element;
      host_gpu_graph_state_.offsetData = checked_u32(offsetData_, "offsetData");
      host_gpu_graph_state_.offsetLevel0 = checked_u32(offsetLevel0_, "offsetLevel0");
      host_gpu_graph_state_.label_offset = checked_u32(label_offset_, "label_offset");
      host_gpu_graph_state_.data_size = gpu_data_size;
      host_gpu_graph_state_.maxlevel = maxlevel_;
      host_gpu_graph_state_.mult = static_cast<float>(mult_);
      host_gpu_graph_state_.revSize = static_cast<float>(revSize_);
      host_gpu_graph_state_.vector_dim = gpu_vector_dim;

      // GPU's own runtime states
      cudaCheck(
          cudaMalloc(&host_gpu_graph_state_.level0_links, gpu_max_elements * gpu_size_links_level0),
          "cudaMalloc",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMemset(host_gpu_graph_state_.level0_links, 0, gpu_max_elements * gpu_size_links_level0),
          "cudaMemset",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMalloc(&host_gpu_graph_state_.level_counts, sizeof(uint32_t) * gpu_max_elements),
          "cudaMalloc",
          __FILE__,
          __LINE__);
      cudaMemset(host_gpu_graph_state_.level_counts, 0, sizeof(uint32_t) * gpu_max_elements);
      cudaCheck(
          cudaMalloc(&host_gpu_graph_state_.element_levels, sizeof(uint32_t) * gpu_max_elements),
          "cudaMalloc",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMalloc(&host_gpu_graph_state_.vector_powers, sizeof(float) * gpu_max_elements),
          "cudaMalloc",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMalloc(&host_gpu_graph_state_.half_vector_data, sizeof(half) * gpu_max_elements * gpu_vector_dim),
          "cudaMalloc",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMalloc(&host_gpu_graph_state_.old_vector_fetch_index, sizeof(uint32_t) * LEVEL_SZ_THRES),
          "cudaMalloc",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMalloc(&(host_gpu_graph_state_.link_lists), sizeof(char *) * gpu_max_elements),
          "cudaMalloc",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMemset(host_gpu_graph_state_.link_lists, 0, sizeof(char *) * gpu_max_elements),
          "cudaMemset",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMalloc(&(host_gpu_graph_state_.visited), sizeof(uint32_t) * gpu_max_elements * GRID_DIM),
          "cudaMalloc",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMemset(host_gpu_graph_state_.visited, 0, sizeof(uint32_t) * gpu_max_elements * GRID_DIM),
          "cudaMemset",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMalloc(&host_gpu_graph_state_.frozen_link_counts, sizeof(uint32_t) * MAX_HNSW_LEVEL * gpu_max_elements),
          "cudaMalloc",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMemset(host_gpu_graph_state_.frozen_link_counts, 0, sizeof(uint32_t) * MAX_HNSW_LEVEL * gpu_max_elements),
          "cudaMemset",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMalloc(
              &host_gpu_graph_state_.changed_old_links,
              sizeof(uint32_t) * static_cast<size_t>(BATCHSZ_PER_NEW) * MAX_HNSW_LEVEL * static_cast<size_t>(maxM0_)),
          "cudaMalloc",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMalloc(&host_gpu_graph_state_.changed_old_link_counts, sizeof(uint32_t) * MAX_HNSW_LEVEL),
          "cudaMalloc",
          __FILE__,
          __LINE__);
      cudaCheck(
          cudaMemset(host_gpu_graph_state_.changed_old_link_counts, 0, sizeof(uint32_t) * MAX_HNSW_LEVEL),
          "cudaMemset",
          __FILE__,
          __LINE__);

      if (old_new_distances_count == 0) {
        host_gpu_graph_state_.news_dist = nullptr;
        host_gpu_graph_state_.news_rank = nullptr;
      } else {
        cudaCheck(
            cudaMalloc(&(host_gpu_graph_state_.news_dist), sizeof(float) * old_new_distances_count),
            "cudaMalloc",
            __FILE__,
            __LINE__);
        cudaCheck(
            cudaMalloc(&(host_gpu_graph_state_.news_rank), sizeof(uint32_t) * old_new_distances_count),
            "cudaMalloc",
            __FILE__,
            __LINE__);
      }

      if (precomputed_new_new_count == 0) {
        host_gpu_graph_state_.precomputed_new_new_dist = nullptr;
        host_gpu_graph_state_.precomputed_new_new_rank = nullptr;
      } else {
        cudaCheck(
            cudaMalloc(&(host_gpu_graph_state_.precomputed_new_new_dist), sizeof(float) * precomputed_new_new_count),
            "cudaMalloc",
            __FILE__,
            __LINE__);
        cudaCheck(
            cudaMalloc(&(host_gpu_graph_state_.precomputed_new_new_rank), sizeof(uint32_t) * precomputed_new_new_count),
            "cudaMalloc",
            __FILE__,
            __LINE__);
      }

      // copy the graph state to GPU
      cudaCheck(cudaMalloc(&device_gpu_graph_state_, sizeof(GpuGraphState)), "cudaMalloc", __FILE__, __LINE__);
      cudaCheck(
          cudaMemcpy(device_gpu_graph_state_, &host_gpu_graph_state_, sizeof(GpuGraphState), cudaMemcpyHostToDevice),
          "cudaMemcpy",
          __FILE__,
          __LINE__);
      cudaCheck(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

      gpu_free_fn_ = &cuda_free_buffer;
      gpu_graph_ready_ = true;
    } catch (...) {
      gpu_free_fn_ = &cuda_free_buffer;
      release_gpu_state();
      throw;
    }
  }

  void copy_from_gpu() {
    printf("Copying graph state from GPU to CPU...\n");
    if (!gpu_graph_ready_ || device_gpu_graph_state_ == nullptr) {
      throw std::runtime_error("copy_from_gpu called before GPU graph state is ready");
    }

    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    cudaCheck(
        cudaMemcpy(&host_gpu_graph_state_, device_gpu_graph_state_, sizeof(GpuGraphState), cudaMemcpyDeviceToHost),
        "cudaMemcpy",
        __FILE__,
        __LINE__);

    const size_t cur_count = host_gpu_graph_state_.cur_element_count;
    if (cur_count > max_elements_) {
      throw std::runtime_error("copy_from_gpu received an invalid cur_element_count");
    }
    if (host_gpu_graph_state_.vector_dim * sizeof(float) != data_size_) {
      throw std::runtime_error("copy_from_gpu received an incompatible vector dimension");
    }

    cur_element_count = cur_count;
    maxlevel_ = host_gpu_graph_state_.maxlevel;
    enterpoint_node_ = cur_count == 0 ? static_cast<tableint>(-1) : 0;

    std::vector<uint32_t> gpu_element_levels(max_elements_, 0);
    cudaCheck(
        cudaMemcpy(
            gpu_element_levels.data(),
            host_gpu_graph_state_.element_levels,
            sizeof(uint32_t) * max_elements_,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy",
        __FILE__,
        __LINE__);
    for (size_t i = 0; i < max_elements_; ++i) {
      if (gpu_element_levels[i] > static_cast<uint32_t>(MAX_HNSW_LEVEL)) {
        throw std::runtime_error("copy_from_gpu received an invalid element level");
      }
      element_levels_[i] = static_cast<int>(gpu_element_levels[i]);
    }

    for (size_t i = 0; i < cur_count; ++i) {
      if (element_levels_[i] == maxlevel_) {
        enterpoint_node_ = static_cast<tableint>(i);
        break;
      }
    }

    std::vector<char *> device_link_lists(max_elements_, nullptr);
    cudaCheck(
        cudaMemcpy(
            device_link_lists.data(),
            host_gpu_graph_state_.link_lists,
            sizeof(char *) * max_elements_,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy",
        __FILE__,
        __LINE__);

    if (host_gpu_graph_state_.level0_links == nullptr || host_gpu_graph_state_.vector_data == nullptr ||
        host_gpu_graph_state_.link_lists == nullptr) {
      throw std::runtime_error("copy_from_gpu received an incomplete GPU graph state");
    }

    auto copy_gpu_link_list = [&](void *cpu_link_list, const char *gpu_link_list, size_t cpu_capacity) {
      if (gpu_link_list == nullptr) {
        throw std::runtime_error("copy_from_gpu found a null GPU link list");
      }

      uint32_t gpu_count = 0;
      cudaCheck(
          cudaMemcpy(&gpu_count, gpu_link_list, sizeof(uint32_t), cudaMemcpyDeviceToHost),
          "cudaMemcpy",
          __FILE__,
          __LINE__);
      if (gpu_count > cpu_capacity) {
        throw std::runtime_error("copy_from_gpu received a GPU link count larger than the CPU list capacity");
      }

      memset(cpu_link_list, 0, sizeof(linklistsizeint) + sizeof(tableint) * cpu_capacity);
      setListCount(static_cast<linklistsizeint *>(cpu_link_list), static_cast<unsigned short int>(gpu_count));
      if (gpu_count == 0) {
        return;
      }

      // GPU layout is: count, id capacity including batch scratch, then distances.
      // CPU layout is only: count, ids. Copy the ids explicitly and skip GPU distances.
      tableint *cpu_ids = reinterpret_cast<tableint *>(static_cast<linklistsizeint *>(cpu_link_list) + 1);
      cudaCheck(
          cudaMemcpy(
              cpu_ids, gpu_link_list + sizeof(linklistsizeint), sizeof(tableint) * gpu_count, cudaMemcpyDeviceToHost),
          "cudaMemcpy",
          __FILE__,
          __LINE__);
    };

    label_lookup_.clear();
    for (size_t i = 0; i < cur_count; ++i) {
      if (linkLists_[i] != nullptr) {
        free(linkLists_[i]);
        linkLists_[i] = nullptr;
      }

      memset(data_level0_memory_ + i * size_data_per_element_, 0, size_data_per_element_);

      copy_gpu_link_list(
          get_linklist0(static_cast<tableint>(i)),
          host_gpu_graph_state_.level0_links + i * host_gpu_graph_state_.size_links_level0,
          maxM0_);

      cudaCheck(
          cudaMemcpy(
              getDataByInternalId(static_cast<tableint>(i)),
              host_gpu_graph_state_.vector_data + i * host_gpu_graph_state_.vector_dim,
              data_size_,
              cudaMemcpyDeviceToHost),
          "cudaMemcpy",
          __FILE__,
          __LINE__);

      const labeltype label = static_cast<labeltype>(i);
      setExternalLabel(static_cast<tableint>(i), label);
      label_lookup_[label] = static_cast<tableint>(i);

      const int level = element_levels_[i];
      if (level <= 0) {
        linkLists_[i] = nullptr;
        continue;
      }

      const size_t cpu_link_list_bytes = size_links_per_element_ * static_cast<size_t>(level);
      linkLists_[i] = (char *)malloc(cpu_link_list_bytes);
      if (linkLists_[i] == nullptr)
        throw std::runtime_error("Not enough memory: copy_from_gpu failed to allocate linklist");
      memset(linkLists_[i], 0, cpu_link_list_bytes);

      for (int lv = 1; lv <= level; ++lv) {
        if (device_link_lists[i] == nullptr) {
          throw std::runtime_error("copy_from_gpu found a null GPU upper-layer link list");
        }
        copy_gpu_link_list(
            linkLists_[i] + static_cast<size_t>(lv - 1) * size_links_per_element_,
            device_link_lists[i] + static_cast<size_t>(lv - 1) * host_gpu_graph_state_.size_links_per_element,
            maxM_);
      }
    }

    for (size_t i = cur_count; i < max_elements_; ++i) {
      if (linkLists_[i] != nullptr) {
        free(linkLists_[i]);
      }
      linkLists_[i] = nullptr;
    }
  }

  void build_graph_gpu() {
    // // Initialize the vectors and the graph state.
    // move_to_gpu();

    cudaEvent_t ev_prepare_start = nullptr;
    cudaEvent_t ev_prepare_end = nullptr;
    cudaEvent_t ev_alloc_start = nullptr;
    cudaEvent_t ev_alloc_end = nullptr;
    cudaEvent_t ev_build_start = nullptr;
    cudaEvent_t ev_build_end = nullptr;
    cudaCheck(cudaEventCreate(&ev_prepare_start), "cudaEventCreate", __FILE__, __LINE__);
    cudaCheck(cudaEventCreate(&ev_prepare_end), "cudaEventCreate", __FILE__, __LINE__);
    cudaCheck(cudaEventCreate(&ev_alloc_start), "cudaEventCreate", __FILE__, __LINE__);
    cudaCheck(cudaEventCreate(&ev_alloc_end), "cudaEventCreate", __FILE__, __LINE__);
    cudaCheck(cudaEventCreate(&ev_build_start), "cudaEventCreate", __FILE__, __LINE__);
    cudaCheck(cudaEventCreate(&ev_build_end), "cudaEventCreate", __FILE__, __LINE__);

    // compute the powers, generate the random levels, and convert to half precision
    cudaCheck(cudaEventRecord(ev_prepare_start), "cudaEventRecord", __FILE__, __LINE__);
    cudaCheck(launch_prepare_graph_kernel(device_gpu_graph_state_), "launch_prepare_graph_kernel", __FILE__, __LINE__);
    cudaCheck(
        launch_precompute_new_new_dist_kernel(device_gpu_graph_state_, host_gpu_graph_state_.max_elements),
        "launch_precompute_new_new_dist_kernel",
        __FILE__,
        __LINE__);
    cudaCheck(cudaEventRecord(ev_prepare_end), "cudaEventRecord", __FILE__, __LINE__);
    cudaCheck(cudaEventSynchronize(ev_prepare_end), "cudaEventSynchronize", __FILE__, __LINE__);
    float ms_prepare = 0.0f;
    cudaCheck(
        cudaEventElapsedTime(&ms_prepare, ev_prepare_start, ev_prepare_end),
        "cudaEventElapsedTime",
        __FILE__,
        __LINE__);

    // assign the link lists for GPU (D2H + cudaMalloc/cudaMemset + pointer table H2D)
    cudaCheck(cudaEventRecord(ev_alloc_start), "cudaEventRecord", __FILE__, __LINE__);
    cudaCheck(
        cudaMemcpy(
            element_levels_.data(),
            host_gpu_graph_state_.element_levels,
            sizeof(int32_t) * element_levels_.size(),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy",
        __FILE__,
        __LINE__);

    std::vector<char *> link_lists(max_elements_);
    for (size_t i = 0; i < max_elements_; ++i) {
      // store the ids and the distances
      int sz = host_gpu_graph_state_.size_links_per_element * element_levels_[i];
      cudaCheck(cudaMalloc((&link_lists[i]), sz), "cudaMalloc", __FILE__, __LINE__);
      cudaCheck(cudaMemset(link_lists[i], 0, sz), "cudaMemset", __FILE__, __LINE__);
    }

    cudaCheck(
        cudaMemcpy(
            host_gpu_graph_state_.link_lists,
            link_lists.data(),
            sizeof(char *) * max_elements_,
            cudaMemcpyHostToDevice),
        "cudaMemcpy",
        __FILE__,
        __LINE__);
    cudaCheck(cudaEventRecord(ev_alloc_end), "cudaEventRecord", __FILE__, __LINE__);
    cudaCheck(cudaEventSynchronize(ev_alloc_end), "cudaEventSynchronize", __FILE__, __LINE__);
    float ms_alloc = 0.0f;
    cudaCheck(
        cudaEventElapsedTime(&ms_alloc, ev_alloc_start, ev_alloc_end),
        "cudaEventElapsedTime",
        __FILE__,
        __LINE__);

    cudaCheck(cudaEventRecord(ev_build_start), "cudaEventRecord", __FILE__, __LINE__);
    cudaCheck(launch_build_graph_kernel(device_gpu_graph_state_), "launch_build_graph_kernel", __FILE__, __LINE__);
    cudaCheck(cudaEventRecord(ev_build_end), "cudaEventRecord", __FILE__, __LINE__);
    cudaCheck(cudaEventSynchronize(ev_build_end), "cudaEventSynchronize", __FILE__, __LINE__);
    float ms_build = 0.0f;
    cudaCheck(
        cudaEventElapsedTime(&ms_build, ev_build_start, ev_build_end), "cudaEventElapsedTime", __FILE__, __LINE__);

    cudaCheck(cudaEventDestroy(ev_prepare_start), "cudaEventDestroy", __FILE__, __LINE__);
    cudaCheck(cudaEventDestroy(ev_prepare_end), "cudaEventDestroy", __FILE__, __LINE__);
    cudaCheck(cudaEventDestroy(ev_alloc_start), "cudaEventDestroy", __FILE__, __LINE__);
    cudaCheck(cudaEventDestroy(ev_alloc_end), "cudaEventDestroy", __FILE__, __LINE__);
    cudaCheck(cudaEventDestroy(ev_build_start), "cudaEventDestroy", __FILE__, __LINE__);
    cudaCheck(cudaEventDestroy(ev_build_end), "cudaEventDestroy", __FILE__, __LINE__);

    fmt::print(
        "build_graph_gpu timing: "
        "prepare_graph_kernel {:.3f} ms, "
        "link_list alloc/setup {:.3f} ms, "
        "build_graph_kernel {:.3f} ms\n",
        ms_prepare,
        ms_alloc,
        ms_build);

    // // copy back to CPU
    // cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    // copy_from_gpu();
  }
};
}  // namespace hnswlib
