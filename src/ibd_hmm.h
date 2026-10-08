#ifndef __IBD_HMM_H
#define __IBD_HMM_H

#include "ibd_utils.h"
#include "ibd_partitions.h"
#include "ibd_emissions.h"
#include "cmd_line.h"

namespace ibd {

class ibd_settings {
  public:
    double two_N;
    double m;
    int mc_samples;
    uint64_t seed;
    int max_coal;
    int max_states;
    bool symmetrize;
    int force_method;
    bool genotype;
    bool fixed;
    bool full_posterior;
    std::string write_q;
    double cache_mb;
    int precision;
    bool verbose;
  
    ibd_settings() : two_N(2e4), m(0.5), mc_samples(1000), seed(216769420), max_coal(-1), max_states(std::numeric_limits<int>::max()),
        symmetrize(true), force_method(-1), genotype(false), fixed(false), full_posterior(false),
        write_q(""), cache_mb(2048), precision(6), verbose(true) {}
  
    ibd_settings(const cmd_line &opt)
      : two_N(opt.ne),
        m(opt.ancestry_proportion[1]),
        mc_samples(opt.ibd_mc),
        seed(static_cast<uint64_t>(opt.ibd_seed)),
        max_coal(opt.ibd_max_coal),
        max_states(opt.ibd_max_states),
        symmetrize(!opt.ibd_no_symmetrize),
        force_method(opt.ibd_method),
        genotype(opt.genotype),
        fixed(opt.ancestral_fixed),
        full_posterior(opt.ibd_full_posterior),
        write_q(opt.ibd_write_q),
        cache_mb(opt.ibd_cache_mb),
        precision(opt.ibd_precision),
        verbose(true) {}
};

class ploidy_group {
public:
  int ploidy;
  state_space sp;
  emission_model em;
  std::vector<int> samples;
  // need to add transition matrix and stuff
  bool cached;
  std::vector<std::vector<arma::mat>> ecache;             // emission cache: [block][member] S x Lb
  std::vector<std::vector<std::vector<double>>> ccache;   // emission log scale cache: [block][member][site]
  ploidy_group() : ploidy(0), cached(false) {}
};
  

class ibd_model {
  public:
   const ibd_data *data;
   ibd_settings set;
   std::vector<std::string> sample_prefix;
   std::vector<int> sample_ploidy;
   std::vector<ploidy_group> groups;
   double current_t;
   long n_bad_sites;

   ibd_model() : data(nullptr), current_t(-1), n_bad_sites(0) {}

   void setup(const ibd_data &data, const ibd_settings &set, const std::vector<markov_chain> &markov_chain_information);
   void block_emissions(ploidy_group &group, size_t block_index, size_t member_index, arma::mat& emission, std::vector<double>& log_scalers);
};

void ibd_model::setup(const ibd_data &data, const ibd_settings &set, const std::vector<markov_chain> &markov_chain_information) {
  this->data = &data;
  this->set = set;
  
  for (const auto &info : markov_chain_information) {
    sample_prefix.push_back(info.output_file);
    sample_ploidy.push_back(int(std::floor(info.number_chromosomes + 0.5)));
  }

  // Group samples by ploidy
  std::vector<int> sample_unique_ploidy = sample_ploidy;
  std::sort(sample_unique_ploidy.begin(), sample_unique_ploidy.end());
  sample_unique_ploidy.erase(std::unique(sample_unique_ploidy.begin(), sample_unique_ploidy.end()), sample_unique_ploidy.end());
  groups.assign(sample_unique_ploidy.size(), ploidy_group());
  for (size_t i = 0; i < sample_unique_ploidy.size(); i++) {
    auto& group = groups[i];
    group.ploidy = sample_unique_ploidy[i];
    for (int j = 0; j < (int) sample_ploidy.size(); j++) {
      if (sample_ploidy[j] == group.ploidy) {
        group.samples.push_back(j);
      }
    }

    int max_coal = set.max_coal;
    if (max_coal < 0 && full_state_space_size(group.ploidy) > set.max_states) {
      // truncate genealogy to at most max_coal within-pool coalescences
      max_coal = 0;
      while (max_coal + 1 <= group.ploidy - 1 && truncated_state_space_size(group.ploidy, max_coal + 1) <= (size_t) set.max_states) max_coal++ ;
      std::cerr << "\t[ibd] ploidy " << group.ploidy << ": full state space has " << full_state_space_size(group.ploidy)
                << " states; truncating to at most " << max_coal << " within-pool coalescences (use --ibd-max-coal to change)\n" ;
    }
  }

  // emission cache if it fits
  double need = 0;
  for (const auto& group : groups) {
    need += 8.0 * double(group.sp.size() * data.n_sites() * group.samples.size());
  }
  bool cache = (need / 1048576.0 <= set.cache_mb);
  for (auto& group : groups) {
    group.cached = false;
    if (!cache) continue;
    group.ecache.assign(data.n_blocks(), std::vector<arma::mat>(group.samples.size()));
    group.ccache.assign(data.n_blocks(), std::vector<std::vector<double>>(group.samples.size()));
    for (size_t b = 0; b < data.n_blocks(); b++) {
      for (size_t s = 0; s < group.samples.size(); s++) {
        block_emissions(group, b, s, group.ecache[b][s], group.ccache[b][s]); // construct emission and log scalers for block/chromosome b and sample s
      }
    }
    group.cached = true;
  }

  if (set.verbose) {
    std::cerr << "\t[ibd] emissions " << (cache ? "precomputed" : "computed on the fly") << " (" << need / 1048576.0 << " MB)\n" ;
  }
}

void ibd_model::block_emissions(ploidy_group &group, size_t block_index, size_t member_index, arma::mat& emission, std::vector<double>& log_scalers) {
  const ibd_data& d = *data ;
  size_t v0 = d.block_start[block_index];
  size_t v1 = d.block_start[block_index+1];
  int sample = group.samples[member_index];
  int state_space_size = (int) group.sp.size();
  
  emission.set_size(state_space_size, v1 - v0);
  log_scalers.assign(v1 - v0, 0.0);
  std::vector<std::vector<double>> V;  // intermedaite
  std::vector<double> Lt;              // likelihood 
  site_weights sw;                     // site-specific ancestry weights
  arma::vec e;                         // final emission likelihood
  for (size_t v = v0; v < v1; v++) {
    double cA = d.counts[sample][2*v];
    double ca = d.counts[sample][2*v+1];
    double lmax = set.genotype ? genotype_likelihood(cA, ca, d.err1[v], d.err2[v], group.ploidy, Lt)
                               : read_likelihood(cA, ca, d.err1[v], d.err2[v], group.ploidy, Lt);
    if (lmax == NEG_INF) {
      if (set.genotype && cA + ca > 0) n_bad_sites++;
      emission.col(v - v0).ones();
      continue;
    }
    compute_site_weights(group.em, &d.ref[4*v], set.fixed, sw);
    state_emissions(group.sp, group.em, sw, Lt, e, V);
    emission.col(v - v0) = e;
    log_scalers[v - v0] = lmax;
  }
}
}
#endif