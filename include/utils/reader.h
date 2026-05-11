#pragma once

#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class FVecItrReader {
 private:
  std::ifstream ifs_;
  std::vector<float> curr_;
  bool eof_flag_;

 public:
  FVecItrReader(std::string dataset_path);
  std::vector<float> Next();
  bool HasEnded();
};

class IVecItrReader {
 private:
  std::ifstream ifs_;
  std::vector<uint32_t> curr_;
  bool eof_flag_;

 public:
  IVecItrReader(std::string dataset_path);
  std::vector<uint32_t> Next();
  bool HasEnded();
};

template <typename T>
class AttrReaderToVector {
 private:
  // Let d be the number of the attributes (each being a float), n be the number of
  // entries. Each following content will be in the format of:
  // | attribute 1 of 1st entry | ... | attribute d of 1st entry |
  // |           ...            | ... |           ...            |
  // | attribute 1 of nth entry | ... | attribute d of nth entry |
  std::vector<std::vector<T>> attrs_;
  bool ready_{false};

 public:
  static void GenerateRandomAttrs(std::string to, uint32_t n, uint32_t d, size_t range, int seed = 0);
  AttrReaderToVector(std::string from) {
    uint32_t n, d;
    std::ifstream ifs(from, std::ios::binary);
    if (!ifs.good()) {
      throw "Failed to open file: " + from;
    }

    ifs.read((char *)&d, sizeof(d));
    ifs.seekg(0, std::ios::end);
    size_t fsize = ifs.tellg();
    n = fsize / ((d + 1) * 4);
    ifs.seekg(0, std::ios::beg);

    try {
      attrs_.resize(n);
      for (size_t i = 0; i < n; i++) {
        attrs_[i].resize(d);
        ifs.read((char *)&d, sizeof(d));
        ifs.read((char *)attrs_[i].data(), d * sizeof(T));
      }
    } catch (const std::exception &e) {
      std::cerr << e.what();
    }
    ready_ = true;
  }

  const std::vector<std::vector<T>> &GetAttrs() {
    if (ready_)
      return attrs_;
    else
      throw "Attribute data not ready yet.";
  }
};

template <typename T>
void AttrReaderToVector<T>::GenerateRandomAttrs(std::string to, uint32_t n, uint32_t d, size_t range, int seed) {
  std::ofstream ofs(to);
  std::mt19937 rng;
  rng.seed(seed);
  std::uniform_real_distribution<> distrib_real;
  for (size_t i = 0; i < n; i++) {
    ofs.write((char *)&d, sizeof(d));
    for (size_t j = 0; j < d; j++) {
      T num = static_cast<T>(distrib_real(rng) * range);
      ofs.write((char *)&num, sizeof(num));
    }
  }
}

template <typename T>
class AttrReaderToRaw {
 private:
  // Let d be the number of the attributes (each being a float), n be the number of
  // entries. Each following content will be in the format of:
  // | attribute 1 of 1st entry | ... | attribute d of 1st entry |
  // |           ...            | ... |           ...            |
  // | attribute 1 of nth entry | ... | attribute d of nth entry |
  T *attrs_;
  bool ready_{false};

 public:
  AttrReaderToRaw(std::string from) {
    uint32_t n, d;
    std::ifstream ifs(from, std::ios::binary);
    if (!ifs.good()) {
      throw "Failed to open file: " + from;
    }

    ifs.read((char *)&d, sizeof(d));
    ifs.seekg(0, std::ios::end);
    size_t fsize = ifs.tellg();
    n = fsize / ((d + 1) * 4);
    ifs.seekg(0, std::ios::beg);

    attrs_ = new T[n * d];

    try {
      for (size_t i = 0; i < n; i++) {
        ifs.read((char *)&d, sizeof(d));
        ifs.read((char *)(attrs_ + i * d), d * sizeof(T));
      }
    } catch (const std::exception &e) {
      std::cerr << e.what();
    }
    ready_ = true;
  }

  T *GetAttrs() {
    if (ready_)
      return attrs_;
    else
      throw "Attribute data not ready yet.";
  }
};
