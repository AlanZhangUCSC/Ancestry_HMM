#ifndef __IBD_UTILS_H
#define __IBD_UTILS_H


#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace ibd {

inline void error(const std::string &msg) {
  std::cerr << "\n\n\t\tERROR (--ibd): " << msg << "\n\n";
  exit(1);
}

const double NEG_INF = -std::numeric_limits<double>::infinity();

// log(exp(a) + exp(b))
inline double log_add(double a, double b) {
  if (a == NEG_INF) return b;
  if (b == NEG_INF) return a;

  const double max = std::max(a, b);
  const double min = std::min(a, b);

  return max + std::log1p(std::exp(min - max));
  // return std::log(std::exp(a) + std::exp(b));
}

struct log_accumulator {
  double max;
  double s;

  log_accumulator() : max(NEG_INF), s(0.0) {}

  void add(double x) {
    if (x == NEG_INF) return;
    if (x <= max) {
      s += std::exp(x - max);
    } else {
      s = s * std::exp(max - x) + 1.0;
      max = x;
    }
  }

  double value() const {
    if (max == NEG_INF) return NEG_INF;
    return max + std::log(s);
  }
};

inline std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    result.push_back(item);
  }
  return result;
}

inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

struct vec_hash {
  size_t operator()(const std::vector<int> &v) const {
      uint64_t h = 1469598103934665603ULL;
      for (size_t i = 0; i < v.size(); i ++) {
          h ^= (uint64_t)(v[i] + 0x9e3779b9) + (h << 6) + (h >> 2);
          h = splitmix64(h);
      }
      return (size_t) h;
  }
};

// log of the binomial coefficient
inline double log_choose(double n, double k) {
  if (k < 0 || k > n) return NEG_INF;
  return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

}

#endif