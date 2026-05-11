#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include "../hnswlib/hnswlib.h"
#include "roaring/roaring.hh"

using namespace hnswlib;
using std::vector;

template <typename attr_t>
class RangeQuery : public hnswlib::BaseFilterFunctor {
 private:
  const attr_t *l_bound_, *u_bound_;  // of dimension d_
  const attr_t *attrs_;               // index should be the labeltype
  size_t n_, d_;

 public:
  RangeQuery(const attr_t *l_bound, const attr_t *u_bound, const attr_t *attrs, size_t n, size_t d)
      : l_bound_(l_bound), u_bound_(u_bound), attrs_(attrs), n_(n), d_(d) {}
  bool operator()(hnswlib::labeltype label) override {
    if (label < n_) {
      for (int i = 0; i < d_; i++) {
        if (l_bound_[i] > attrs_[label * d_ + i] || attrs_[label * d_ + i] > u_bound_[i]) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  bool operator()(const std::array<attr_t, 4> &attrs) {
    for (int i = 0; i < d_; i++) {
      if (l_bound_[i] > attrs[i] || attrs[i] > u_bound_[i]) {
        return false;
      }
    }
    return true;
  }

  const attr_t *prefetch(labeltype label) { return attrs_ + label * d_; }
};

template <typename attr_t>
class WindowQuery : public hnswlib::BaseFilterFunctor {
 private:
  const attr_t *l_bound_, *u_bound_;  // of the same dimension as attrs

 public:
  WindowQuery(const attr_t *l_bound, const attr_t *u_bound) : l_bound_(l_bound), u_bound_(u_bound) {}
  bool operator()(const std::vector<attr_t> &attrs) {
    for (int i = 0; i < attrs.size(); i++) {
      if (l_bound_[i] > attrs[i] || attrs[i] > u_bound_[i]) {
        return false;
      }
    }
    return true;
  }
};

template <typename attr_t, typename dist_t>
class RangedQueryStopCondition : public hnswlib::BaseSearchStopCondition<dist_t> {
 private:
  attr_t l_bound_, u_bound_;
  vector<attr_t> *attrs_;
  size_t curr_num_items_;
  vector<std::pair<dist_t, labeltype>> temp_result_;
  size_t window_size_;

 public:
  RangedQueryStopCondition(attr_t l_bound, attr_t u_bound, const vector<attr_t> *attrs)
      : l_bound_(l_bound), u_bound_(u_bound), attrs_(attrs) {
    curr_num_items_ = 0;
  }

  void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) override {
    curr_num_items_ += 1;
    temp_result_.emplace(dist, label);
  }

  bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) override {
    bool stop_search = true;
    size_t i = 0;
    for (auto it = temp_result_.rbegin(); it != temp_result_.rend() && i < window_size_; it++, i++) {
      auto attr_value = attrs_[it.second];
      if (l_bound_ <= attr_value && attr_value <= u_bound_) {
        stop_search = false;
        break;
      }
    }
    return stop_search;
  }
};

template <typename attr_t>
class BitsetQuery : public hnswlib::BaseFilterFunctor {
 private:
  roaring::Roaring bitset;
  size_t n_, d_;
  bool flipped_;

 public:
  BitsetQuery(const attr_t *l_bound, const attr_t *u_bound, const attr_t *attrs, size_t n, size_t d)
      : n_(n), d_(d), flipped_(false) {
    for (int i = 0; i < n; i++) {
      bool ok = true;
      for (int j = 0; j < d; j++) {
        if (attrs[i * d + j] < l_bound[j] || attrs[i * d + j] > u_bound[j]) {
          ok = false;
          break;
        }
      }
      if (ok) {
        bitset.add(i);
      }
    }
    bitset.runOptimize();
    if (bitset.cardinality() < 100000) {
      bitset.flip(0, bitset.maximum() + 1);
      flipped_ = true;
    }
  }
  bool operator()(hnswlib::labeltype label) override {
    if (label < n_) {
      return flipped_ ? !bitset.contains(label) : bitset.contains(label);
    }
    return false;
  }
};

template <typename attr_t>
class InplaceRangeQuery : public hnswlib::BaseFilterFunctor {
 private:
  const attr_t l_range_, u_range_;
  const attr_t *l_bound_, *u_bound_;
  const attr_t *attrs_;
  size_t n_, d_;

 public:
  InplaceRangeQuery(const attr_t l_range, const attr_t u_range, size_t n, size_t d)
      : l_range_(l_range), u_range_(u_range), n_(n), d_(d) {}
  InplaceRangeQuery(
      const attr_t l_range,
      const attr_t u_range,
      const attr_t *l_bound,
      const attr_t *u_bound,
      const attr_t *attrs,
      size_t n,
      size_t d
  )
      : l_range_(l_range), u_range_(u_range), l_bound_(l_bound), u_bound_(u_bound), attrs_(attrs), n_(n), d_(d) {}
  bool operator()(hnswlib::labeltype label) override {
    if (label < l_range_ || label > u_range_) {
      return false;
    }
    for (int i = 1; i < d_; i++) {
      if (attrs_[label * d_ + i] < l_bound_[i - 1] || attrs_[label * d_ + i] > u_bound_[i - 1]) {
        return false;
      }
    }
    return true;
  }

  const attr_t* prefetch(labeltype label) { return attrs_ + label * d_; }
};