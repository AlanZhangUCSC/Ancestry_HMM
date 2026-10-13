#ifndef __IBD_RATE_MATRIX_H
#define __IBD_RATE_MATRIX_H

#include "ibd_utils.h"
#include "ibd_partitions.h"
#include "ibd_genealogy.h"

namespace ibd {


class rate_entry {
public:
  int from_state;
  int to_state;
  double transition_rate;

  rate_entry(int from_state_, int to_state_, double transition_rate_) : from_state(from_state_), to_state(to_state_), transition_rate(transition_rate_) {}
};

class rate_model {
public:
  int n; // ploidy
  double t;
  double two_N;
  double tau;
  double m;
  int mc_samples;
  uint64_t seed;
  bool symmetrized;

  std::vector<double> logpi;      // logpi renormalized
  std::vector<double> logpi_full; // untruncated log pi
  double retained_mass;           // sum of untruncated pi

  std::vector<partition_kernels> kernels;

  // off-diagonal components (per morgan)
  std::vector<rate_entry> Q_move; // transitions that reattach to another existing class
  std::vector<rate_entry> Q_esc0; // transitions that escape to admixture time and become ancestry 0
  std::vector<rate_entry> Q_esc1; // transitions that escape to admixture time and become ancestry 1

  arma::mat Q_raw; // monte carlo generator
  arma::mat Q;     // generator actually used by the HMM

  // diagnostics
  double identity_error;
  double stationarity_error;
  double detailed_balance_asym;
  double max_rowsum_error;

  void build(const state_space& sp, double t_, double two_N_, double m_, int mc_samples_, uint64_t seed_, bool symmetrize, bool check_identity);
  void write(const state_space& sp, std::ostream& out) const;
};

// log pi(b) under the untruncated coalescent with K classes...
std::vector<double> log_stationary(const state_space &sp, double tau, double m) {
  const int n = sp.n; // ploidy
  std::vector<double> logPK = log_lineage_distribution(n, tau); // log of distribution of number of lineages
  std::vector<double> log_stationary_probs(sp.size(), NEG_INF);
  double lm0 = (m < 1) ? std::log1p(-m) : NEG_INF; // ancestry 0 prob
  double lm1 = (m > 0) ? std::log(m) : NEG_INF;    // ancestry 1 prob
  for (size_t s = 0; s < sp.size(); s ++) {
      const colored_state &st = sp.states[s];
      int K = st.K0 + st.K1;  // number of IBD classes
      double logp = logPK[K] + std::lgamma(n - K + 1.0) + std::lgamma(K + 1.0) + std::lgamma((double) K) - std::lgamma((double) n);
      for (int c = 1; c <= n; c ++) {
          logp -= std::lgamma(st.b0[c] + 1.0) + std::lgamma(st.b1[c] + 1.0);
      }
      if (st.K0 > 0) logp += st.K0 * lm0;
      if (st.K1 > 0) logp += st.K1 * lm1;
      log_stationary_probs[s] = logp;
  }
  return log_stationary_probs;
}

void rate_model::build(const state_space& sp, double t_, double two_N_, double m_, int mc_samples_, uint64_t seed_, bool symmetrize, bool check_identity) {
  n = sp.n;
  t = t_;
  two_N = two_N_;
  tau = t / two_N;  // admixture time measured in coalescent units, t/2N
  m = m_;
  mc_samples = mc_samples_;
  seed = seed_;
  symmetrized = symmetrize;

  const int num_states = (int) sp.size();

  // stationary distribution
  logpi_full = log_stationary(sp, tau, m);
  logpi.resize(num_states);

  log_accumulator acc;
  for (int i = 0; i < num_states; i++) acc.add(logpi_full[i]);
  double log_retained_mass = acc.value();
  retained_mass = std::exp(log_retained_mass);
  for (int i = 0; i < num_states; i++) logpi[i] = logpi_full[i] - log_retained_mass;

  // monte carlo kernels per uncolored partition
  compute_partition_kernels(sp, tau, mc_samples, seed, check_identity, kernels, identity_error);

  // Now colored partitions....
  Q_move.clear();
  Q_esc0.clear();
  Q_esc1.clear();
  std::vector<int> next_b0;
  std::vector<int> next_b1;
  for (int i = 0; i < num_states; i++) {
    const colored_state& source_state = sp.states[i];
    const partition_kernels& kernel = kernels[source_state.part];
    const std::vector<int> *b[2] = {&source_state.b0, &source_state.b1};

    for (int j = 0; j < 2; j++) {      // j is ancestry index (0 or 1)
      for (int c = 1; c <= n; c++) {   // c is every possible source IBD-class size
        int bjc = (*b[j])[c];          // number of IBD-classes of ancestry j with size c. Same notation as in the manuscript.
        if (bjc == 0) continue;
        
        for (int k = 1; k <= c; k++) { // moving k chromosome copies out of a class of size c

          // case 1: moves into a distinct class (j2 , c2)
          for (int j2 = 0; j2 < 2; j2++) { // j2 is target ancestry index (0 or 1)
            for (int c2 = 1; c2 <= n - c; c2++) { // c2 is target IBD-class size
              int num_transition_pairs = bjc * ((*b[j2])[c2] - ((j2 == j && c2 == c) ? 1 : 0));
              if (num_transition_pairs == 0) continue;
              
              double move_rate = kernel.get_move_rate(c, c2, k);
              if (move_rate <= 0) continue;

              next_b0 = source_state.b0;
              next_b1 = source_state.b1;
              std::vector<int> *next_b[2] = {&next_b0, &next_b1};

              (*next_b[j])[c]--;
              (*next_b[j2])[c2]--;
              (*next_b[j2])[c2+k]++;
              if (k < c) (*next_b[j])[c-k]++;
              if (next_b0 == source_state.b0 && next_b1 == source_state.b1) continue;
              int i2 = sp.find(next_b0, next_b1);
              if (i2 < 0) continue; // truncated state space
              Q_move.push_back(rate_entry(i, i2, two_N * num_transition_pairs * move_rate));
            }
          }

          // case 2: no-coalescence before admixture time, new ancestry jstar
          double esc_rate = kernel.get_esc_rate(c, k);
          if (esc_rate <= 0) continue;
          for (int jstar = 0; jstar < 2; jstar++) {
            next_b0 = source_state.b0;
            next_b1 = source_state.b1;
            std::vector<int> *next_b[2] = {&next_b0, &next_b1};

            (*next_b[j])[c]--;
            (*next_b[jstar])[k]++;
            if (k < c) (*next_b[j])[c-k]++;
            if (next_b0 == source_state.b0 && next_b1 == source_state.b1) continue;
            int i2 = sp.find(next_b0, next_b1);
            if (i2 < 0) continue; // truncated state space
            if (jstar == 0) Q_esc0.push_back(rate_entry(i, i2, two_N * bjc * esc_rate));
            else            Q_esc1.push_back(rate_entry(i, i2, two_N * bjc * esc_rate));
          }
        }
      }
    }
  }

  // Q(m) = Q_move + (1-m) Q_esc0 + m Q_esc1
  Q_raw.zeros(num_states, num_states);
  for (size_t i = 0; i < Q_move.size(); i++) Q_raw(Q_move[i].from_state, Q_move[i].to_state) += Q_move[i].transition_rate;
  for (size_t i = 0; i < Q_esc0.size(); i++) Q_raw(Q_esc0[i].from_state, Q_esc0[i].to_state) += Q_esc0[i].transition_rate * (1 - m);
  for (size_t i = 0; i < Q_esc1.size(); i++) Q_raw(Q_esc1[i].from_state, Q_esc1[i].to_state) += Q_esc1[i].transition_rate * m;

  // set diagonal to 0 and set diagonal to -rowsum
  for (int i = 0; i < num_states; i++) {
    Q_raw(i, i) = 0.0;
    Q_raw(i, i) = -arma::accu(Q_raw.row(i));
  }

  // sanity check for monte carlo
  {
    arma::vec pi(num_states); // stationary prob
    for (int i = 0; i < num_states; i++) pi(i) = std::exp(logpi[i]);
    arma::rowvec r = pi.t() * Q_raw; // residual

    double denom = 0;
    for (int i = 0; i < num_states; i++) denom += pi(i) * std::fabs(Q_raw(i,i));
    stationarity_error = (denom > 0) ? arma::accu(arma::abs(r)) / denom : 0.0;

    double num = 0;
    double den = 0;
    for (int i = 0; i < num_states; i++) {
      for (int j = i + 1; j < num_states; j++) {
        double forward = pi(i) * Q_raw(i,j);
        double reverse = pi(j) * Q_raw(j,i);
        num += std::fabs(forward - reverse);
        den += forward + reverse;
      }
    }
    detailed_balance_asym = (den > 0) ? num / den : 0.0;
    
    max_rowsum_error = 0;
    for (int i = 0; i < num_states; i++) {
      double scale = std::max(1e-300, std::fabs(Q_raw(i,i)));
      max_rowsum_error = std::max(max_rowsum_error, std::fabs(arma::accu(Q_raw.row(i))) / scale);
    }
  }

  if (!symmetrized) {
    Q = Q_raw;
    return;
  }

  // symmetrize Q
  Q.zeros(num_states, num_states);
  for (int i = 0; i < num_states; i++) {
    for (int j = i + 1; j < num_states; j++) {
      double forward = Q_raw(i,j);
      double reverse = Q_raw(j,i);
      if (forward <= 0 && reverse <= 0) continue;

      double log_forward = (forward > 0) ? std::log(forward) : NEG_INF;
      double log_reverse = (reverse > 0) ? std::log(reverse) : NEG_INF;

      double delta = logpi[j] - logpi[i];
      double reverseible_forward = std::exp(log_add(log_forward, (log_reverse == NEG_INF) ? NEG_INF : log_reverse + delta)) * 0.5;
      double reversible_reverse  = std::exp(log_add(log_reverse, (log_forward == NEG_INF) ? NEG_INF : log_forward - delta)) * 0.5;
      if (!std::isfinite(reverseible_forward) || !std::isfinite(reversible_reverse)) {
        error("Non-finite rate in the reversible projection of Q");
      }

      Q(i,j) = reverseible_forward;
      Q(j,i) = reversible_reverse;
    }
  }

  // set diagonal to 0 and set diagonal to -rowsum again
  for (int i = 0; i < num_states; i++) {
    Q(i,i) = 0.0;
    Q(i,i) = -arma::accu(Q.row(i));
  }
}

void rate_model::write(const state_space& sp, std::ostream& out) const {
  const int stats_space_size = (int) sp.size();
  out.precision(12);
  out << "#ancestry_ibd_rate_matrix\n";
  out << "#n\t" << n << "\n#t\t" << t << "\n#two_N\t" << two_N << "\n#tau\t" << tau << "\n#m\t" << m << "\n";
  out << "#mc_samples\t" << mc_samples << "\n#seed\t" << seed << "\n#max_coal\t" << sp.max_coal << "\n";
  out << "#states\t" << stats_space_size << "\n#full_state_space\t" << sp.full_size << "\n#retained_pi_mass\t" << retained_mass << "\n";
  out << "#identity_error\t" << identity_error << "\n#stationarity_error_raw\t" << stationarity_error << "\n";
  out << "#detailed_balance_asymmetry_raw\t" << detailed_balance_asym << "\n#symmetrized\t" << (symmetrized ? 1 : 0) << "\n";
  out << "STATE\tindex\tlabel\tN0\tN1\tK0\tK1\tlog_pi\n";
  for (int s = 0; s < stats_space_size; s ++) {
    const colored_state &st = sp.states[s];
    out << "STATE\t" << s << "\t" << st.label << "\t" << st.N0 << "\t" << st.N1 << "\t" << st.K0 << "\t" << st.K1 << "\t" << logpi[s] << "\n";
  }
  out << "RATE\tfrom\tto\tQ_raw\tQ_used\n";
  for (int s = 0; s < stats_space_size; s ++) {
    for (int s2 = 0; s2 < stats_space_size; s2 ++) {
      if (s == s2) continue;
      if (Q_raw(s, s2) != 0 || Q(s, s2) != 0) {
        out << "RATE\t" << s << "\t" << s2 << "\t" << Q_raw(s, s2) << "\t" << Q(s, s2) << "\n";
      }
    }
  }
}
  
}

#endif