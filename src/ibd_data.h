#ifndef __IBD_DATA_H
#define __IBD_DATA_H

/*
 Input for the ancestry-IBD HMM.  Same file format as Ancestry_HMM:
   chrom  pos  [C_A C_a for each ancestral population]  distance(Morgans)
   [error_1 error_2  if -E]  [A a counts for each sample]
 Unlike read_input.h, the raw counts are stored and emissions are computed
 later (they depend on the IBD state space of each sample's ploidy).

 Behaviour kept from read_input.h: duplicate positions are skipped, sites
 closer than -d Morgans to the previous retained site are skipped (their
 distance is accumulated), and each chromosome is an independent block that
 starts from the stationary distribution.  Not kept: random down-sampling of
 reads above depth 170 and of reference panels above 998 chromosomes (only
 needed by the factorial-based emissions of the original code).
*/

#include "ibd_utils.h"
#include <fstream>

namespace ibd {

class ibd_data {
public:
  std::vector<std::string> chrom;
  std::vector<int> pos;
  std::vector<double> dist;               // Morgans from the previous retained site of the same chromosome
  std::vector<size_t> block_start;        // first site of each chromosome, plus a final sentinel = n_sites
  std::vector<double> ref;                // 4 per site for 2 ancestry populations: C0A, C0a, C1A, C1a
  std::vector<double> err1, err2;
  std::vector<std::vector<double> > counts;   /// counts[sample][2*site + {0,1}] = (A, a)

  size_t n_sites() const { return pos.size(); }
  size_t n_blocks() const { return block_start.empty() ? 0 : block_start.size() - 1; }
};

inline void read_ibd_input(
  const std::string &file,
  int n_ancestry,
  int n_samples,
  double minimum_distance,
  bool site_errors,
  double error_rate,
  ibd_data &data
) {
  if (n_ancestry != 2) {
    error("IBD mode currently only supports 2 ancestry populations");
  }

  std::ifstream in(file.c_str());
  if (!in) error("cannot open input file " + file);

  data.counts.assign(n_samples, std::vector<double>());
  
  std::string line;
  std::string last_chrom;
  double extra = 0.0;
  size_t line_no = 0;

  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty()) continue;

    std::vector<std::string> fields = split(line, '\t');
  
    size_t i = 0;
    std::string ch = fields[i++];
    int p = std::stoi(fields[i++]);
  
    std::vector<double> ref_counts(2 * n_ancestry);
    for (double &ref_count : ref_counts) {
      ref_count = std::stod(fields[i++]);
    }
  
    double morgan = std::stod(fields[i++]);
    double e1 = error_rate;
    double e2 = error_rate;
    if (site_errors) {
      e1 = std::stod(fields[i++]);
      e2 = std::stod(fields[i++]);
    }
  
    std::vector<double> sample_counts(2 * n_samples);
    for (double &sample_count : sample_counts) {
      sample_count = std::stod(fields[i++]);
    }
  
    bool new_chrom = ch != last_chrom;

    // Skip duplicate positions.
    if (!new_chrom && !data.pos.empty() && p == data.pos.back()) {
      std::cerr << "Warning: duplicate position " << p << " on chromosome " << ch << " at line " << line_no << " ... First occurrence is kept." << std::endl;
      continue;
    }

    double distance = 0.0;
    if (new_chrom) {
      last_chrom = ch;
      extra = 0.0;
      data.block_start.push_back(data.pos.size());
    } else {
      extra += morgan;
      if (extra < minimum_distance) continue; // Skip sites closer than minimum_distance to the previous retained site.

      distance = extra;
      extra = 0.0;
    }

    // Add data to the data object.
    data.chrom.push_back(ch);
    data.pos.push_back(p);
    data.dist.push_back(distance);
    data.ref.insert(data.ref.end(), ref_counts.begin(), ref_counts.end());
    data.err1.push_back(e1);
    data.err2.push_back(e2);
    for (int s = 0; s < n_samples; s ++) {
      data.counts[s].push_back(sample_counts[2 * s]);
      data.counts[s].push_back(sample_counts[2 * s+1]);
    }
  }
  data.block_start.push_back(data.pos.size());
  if (data.pos.empty()) {
    error("no sites read from " + file);
  }
}

}

#endif
