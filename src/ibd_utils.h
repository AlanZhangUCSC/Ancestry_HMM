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
  const double NEG_INF = -std::numeric_limits<double>::infinity();

  inline void error(const std::string &msg) {
    std::cerr << "\n\n\t\tERROR (--ibd): " << msg << "\n\n" ;
    exit(1) ;
  }

  inline std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
      result.push_back(item);
    }
    return result;
  }
}

#endif