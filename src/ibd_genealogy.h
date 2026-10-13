#ifndef __IBD_GENEALOGY_H
#define __IBD_GENEALOGY_H

#include "ibd_utils.h"
#include "ibd_partitions.h"
#include "ibd_death_process.h"
#include <random>

namespace ibd {

class partition_kernels {
public:
  std::vector<int> multiplicities;  // nu => nu[c] => number of IBD classes of size c
  std::vector<int> sizes;
  std::vector<int> indices;         // indices[c] => index of size c in sizes, -1 if not present

  int cmax;
  std::vector<double> move_rates;   // flattened 3d table
  std::vector<double> esc_rates;    // flattened 2d table
  double max_identity_error;

  void init(const std::vector<int> &multiplicities_) {
    multiplicities = multiplicities_;
    int n = (int) multiplicities.size() - 1;

    cmax = 0;
    indices.assign(n + 1, -1);
    for (int c = 1; c <= n; c++) {
      if (multiplicities[c] > 0) {
        indices[c] = (int) sizes.size();
        sizes.push_back(c);
        cmax = c;
      }
    }

    size_t num_sizes = sizes.size();
    move_rates.assign(num_sizes * num_sizes * (cmax + 1), 0.0);  // source size x target size x chrom copies
    esc_rates.assign(num_sizes * (cmax + 1), 0.0);               // source size x chrom copies
    max_identity_error = 0.0;
  }

  double& move_rate_ref(int c, int c2, int d) {
    size_t m = sizes.size();
    return move_rates[ ((size_t) indices[c] * m + indices[c2])
                     * (cmax + 1)
                     + d]; // flattens 3d index
  }

  double& esc_rate_ref(int c, int d) {
    return esc_rates[ (size_t) indices[c]
                * (cmax + 1)
                + d]; // flattens 2d index
  }

  double get_move_rate(int c, int c2, int d) const {
    // skip over invalid cases
    if (c < 1 || c2 < 1 || c >= (int) indices.size() || c2 >= (int) indices.size() || indices[c] < 0 || indices[c2] < 0 || d < 1 || d > c) return 0.0;
    
    size_t m = sizes.size();
    return move_rates[ ((size_t) indices[c] * m + indices[c2])
                     * (cmax + 1)
                     + d]; // flattens 3d index
  }

  double get_esc_rate(int c, int d) const {
    // skip over invalid cases
    if (c < 1 || c >= (int) indices.size() || indices[c] < 0 || d < 1 || d > c) return 0.0;
    
    return esc_rates[ (size_t) indices[c]
                    * (cmax + 1)
                    + d]; // flattens 2d index
  }

};

// reusable object for processing sampled genealogies
// Each sampled genealogy contributes the closed-form interval sums
// kappa_move = sum_k l_k [ (a'_k/A_k)(Delta_k - (1-g_k)/A_k) + ((1-g_k)/A_k) F_k ]
// kappa_esc  = sum_k l_k ((1-g_k)/A_k) H_k
// with g_k = exp(-A_k Delta_k), F_k = a'_{k+1}(1-g_{k+1})/A_{k+1} + g_{k+1}F_{k+1},
// H_k = g_{k+1}H_{k+1}, F_last = 0, H_last = 1.
class genealogy_workspace {
public:
  std::vector<double> Delta, A, gam, O, P, H, Y, F, X, aprime;
  std::vector<std::vector<int>> cnt;                 //cnt[h][k]: number of coealescence events belonging to class h occured before interval k
  std::vector<std::vector<std::vector<int>>> hist;   //hist[h][e][d]: In IBD class h, the number of branches that subtend exactly d sampled chromosome copies after e within class coalesence events 
  std::vector<std::vector<double>> Ssize;            //Ssize[i][k]: During interval k, the number of ancestral lineages that belong to original IBD classes whose total class size is sizes[i]
  std::vector<int> ev;
  std::vector<int> cur;
};

template <class RNG>
void accumulate_genealogy(
  partition_kernels& acc,
  const std::vector<int>& class_sizes,
  const double* times,
  double tau,
  RNG& rng,
  genealogy_workspace& w,
  bool check_identity
){
  const int K = class_sizes.size();  // number of IBD classes
  int K2 = 0;                        // number of non-singleton IBD classes
  while (K2 < K && class_sizes[K2] > 1) K2++;
  const int nu1 = K - K2;            // number of singleton IBD classes

  int n = 0; // total chromosome copy count
  for (int i = 0; i < K; i++) n += class_sizes[i];

  const int n_events = n - K;  // number of coalesence events to reduce n sampled chromosomes to the number of classes (K)
  const int n_intervals = n_events + 1;  // number of intervals required, +1 for the final interval to tau.

  // Sample topology for each non-singleton IBD class
  w.hist.resize(K2);
  for (int h = 0; h < K2; h++) { // IBD class h
    int c = class_sizes[h]; // size of class h
    w.hist[h].assign(c, std::vector<int>(c + 1, 0));
    w.cur.assign(c, 1); // initialize to c separate lineages, each having one descendant sampled chromosome copy
    w.hist[h][0][1] = c;  // initialize to c branches each substending one sampled chromosome copy
    for (int e = 1; e < c; e++) {
      int L = (int) w.cur.size();  // active lineages

      // select two random lineages to coalesce
      int i = uniform_index(rng, L);
      int j = uniform_index(rng, L - 1);
      if (j >= i) j++;
      if (i > j) std::swap(i, j);

      // coalesce
      w.cur[i] += w.cur[j]; 
      w.cur.erase(w.cur.begin() + j);

      // update history
      for (size_t l = 0; l < w.cur.size(); l++) w.hist[h][e][w.cur[l]]++;
    }
  }

  // set random chronological order
  w.ev.clear();
  for (int h = 0; h < K2; h++) {
    for (int e = 0; e < class_sizes[h] - 1; e++) {
      w.ev.push_back(h);
    }
  }
  std::shuffle(w.ev.begin(), w.ev.end(), rng);

  // set cnt
  w.cnt.assign(K2, std::vector<int>(n_intervals, 0));
  for (int k = 1; k < n_intervals; k++) {
    for (int h = 0; h < K2; h++) {
      w.cnt[h][k] = w.cnt[h][k-1];
    }
    w.cnt[w.ev[k-1]][k]++;
  }

  // intervals
  w.Delta.resize(n_intervals);
  w.A.resize(n_intervals);
  w.gam.resize(n_intervals);
  w.O.resize(n_intervals);
  w.P.resize(n_intervals);
  w.H.resize(n_intervals);
  w.Y.resize(n_intervals);
  w.F.resize(n_intervals);
  w.X.resize(n_intervals);
  w.aprime.resize(n_intervals);
  for (int k = 0; k < n_intervals; k++) {
    double lo = (k == 0) ? 0.0 : times[k-1];
    double hi = (k == n_events) ? tau : times[k];
    double Dk = hi - lo;
    if (Dk < 0) Dk = 0.0;
    double Ak = (double) (n - k); // active lineages
    double x = Ak * Dk;
    w.Delta[k] = Dk;
    w.A[k] = Ak;
    w.gam[k] = std::exp(-x);
    w.O[k] = -std::expm1(-x) / Ak;
    w.P[k] = phi_x(x) / (Ak * Ak);
  }
  w.H[n_intervals-1] = 1.0;
  for (int k = n_intervals - 2; k >= 0; k--) {
    w.H[k] = w.gam[k+1] * w.H[k+1];
  }
  double sumY = 0; // escape weight
  for (int k = 0; k < n_intervals; k++) {
    w.Y[k] = w.O[k] * w.H[k];
    sumY += w.Y[k];
  }

  // lineages per target size during each interval
  const std::vector<int> &sizes = acc.sizes;
  const int m = sizes.size();
  w.Ssize.assign(m, std::vector<double>(n_intervals, 0.0));
  if (nu1 > 0) {
    for (int k = 0; k < n_intervals; k++) {
      w.Ssize[acc.indices[1]][k] = nu1;
    }
  }
  for (int h = 0; h < K2; h++) {
    int i = acc.indices[class_sizes[h]];
    for (int k = 0; k < n_intervals; k++) {
      w.Ssize[i][k] += class_sizes[h] - w.cnt[h][k];
    }
  }

  // case: escape rate
  if (nu1 > 0) {
    acc.esc_rate_ref(1, 1) += nu1 * sumY;
  }
  for (int h = 0; h < K2; h++) {
    int c = class_sizes[h];
    for (int k = 0; k < n_intervals; k++) {
      const std::vector<int> &hk = w.hist[h][w.cnt[h][k]];
      for (int d = 1; d <= c; d++) {
        if (hk[d]) {
          acc.esc_rate_ref(c, d) += hk[d] * w.Y[k];
        }
      }
    }
  }

  // case: move rate
  for (int i = (nu1 > 0 ? -1 : 0); i < K2; i++) { // source class ... -1 represents the entire group of singleton classes
    int c = (i < 0) ? 1 : class_sizes[i]; // source class size

    // for sanity check
    std::vector<double> ident_lhs, ident_rhs;
    if (check_identity) {
      ident_lhs.assign(c + 1, 0.0);
      ident_rhs.assign(c + 1, 0.0);
      for (int k = 0; k < n_intervals; k++) {
        for (int d = 1; d <= c; d++) {
          double l = (i < 0) ? 1.0 : w.hist[i][w.cnt[i][k]][d];
          ident_rhs[d] += l * w.Delta[k];
          ident_lhs[d] += l * w.Y[k];
        }
      } 
    }

    for (int i2 = -1; i2 < m; i2++) { // target class 
      if (i2 < 0 && !check_identity) continue; // -1 represents the source class itself... used for sanity check
      int c2 = (i2 < 0) ? c : sizes[i2]; // target class size
      
      bool any = false; // any active lineages in target class?
      for (int k = 0; k < n_intervals; k++) {
        double own = (i < 0) ? 1.0 : (double) (c - w.cnt[i][k]); // number of active lineages in current source
        double ap; // eligible target lineages
        
        if (i2 < 0) ap = own;
        else        ap = w.Ssize[i2][k] - (c2 == c ? own : 0.0);

        if (ap < 0.5) ap = 0.0;
        w.aprime[k] = ap;
        if (ap > 0) any = true;
      }
      if (!any) continue;

      w.F[n_intervals-1] = 0.0;
      for (int k = n_intervals - 2; k >= 0; k--) {
        w.F[k] = w.aprime[k+1] * w.O[k+1] + w.gam[k+1] * w.F[k+1];
      }
      for (int k = 0; k < n_intervals; k++) {
        w.X[k] = w.aprime[k] * w.P[k] + w.O[k] * w.F[k];
      }

      if (i < 0) {
        // move contribution from singleton classes
        double sx = 0;
        for (int k = 0; k < n_intervals; k++) sx += w.X[k];
        if (i2 >= 0) acc.move_rate_ref(1, c2, 1) += nu1 * sx;
        if (check_identity) ident_lhs[1] += sx;
      } else {
        // move contribution from non-singleton classes
        for (int k = 0; k < n_intervals; k++) {
          const std::vector<int> &hk = w.hist[i][w.cnt[i][k]];
          for (int d = 1; d <= c; d++) {
            if (!hk[d]) continue;
            double v = hk[d] * w.X[k];
            if (i2 >= 0) acc.move_rate_ref(c, c2, d) += v;
            if (check_identity) ident_lhs[d] += v;
          }
        }
      }
    }

    if (check_identity) {
      for (int d = 1; d <= c; d++) {
        if (ident_rhs[d] <= 0) continue;
        double err = std::fabs(ident_lhs[d] - ident_rhs[d]) / ident_rhs[d];
        if (err > acc.max_identity_error) acc.max_identity_error = err;
      }
    }
  }
}

/// seeds that depend only on (seed, K) and (seed, partition), not on enumeration order
inline uint64_t time_stream_seed(uint64_t seed, int n, int K) {
  return splitmix64(splitmix64(seed ^ 0x5bd1e9955bd1e995ULL) + (uint64_t) n * 1000003ULL + (uint64_t) K);
}
inline uint64_t topology_stream_seed(uint64_t seed, const std::vector<int> &nu) {
  return splitmix64(seed ^ (uint64_t) vec_hash()(nu));
}

// For every retained uncolored partition, estimate its move and escape kernels by averaging over M genealogies
void compute_partition_kernels(
  const state_space &sp,
  double tau,
  int M,
  uint64_t seed,
  bool check_identity,
  std::vector<partition_kernels> &out,
  double &max_identity_error
) {
  const int n  = sp.n;                 // number of chromosomes
  const int np = sp.partitions.size(); // number of partitions

  std::vector<int> needK(n + 1, 0); 
  for (int i = 0; i < np; i++) {
    needK[partition_parts(sp.partitions[i])] = 1; // boolean: 1 if at least one partition has K classes.
  }

  // coalescent times for K..
  std::vector<std::vector<double>> times(n + 1);
  std::vector<int> Ks;
  for (int k = 1; k <= n; k++) {
    if (needK[k]) Ks.push_back(k);
  }

  // sample coalescence times in parallel
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < Ks.size(); i++) {
    int k = Ks[i];
    int n_events = n - k;
    times[k].assign((size_t) M * n_events, 0.0);

    if (n_events == 0) continue;
    death_process d_process;
    d_process.init(n, k, tau);
    std::mt19937_64 rng(time_stream_seed(seed, n, k));
    std::vector<double> u(n_events); //temp
    for (int m = 0; m < M; m++) {
      for (int e = 0; e < n_events; e++) {
        u[e] = open_uniform(rng);
      }
      d_process.sample_times(&u[0], &times[k][(size_t) m * n_events]);
    }
  }

  out.assign(np, partition_kernels());
  std::vector<double> ident(np, 0.0);

  // process partitions in parallel
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < np; i++) {
    const std::vector<int>& nu = sp.partitions[i];
    partition_kernels& accumulator = out[i];
    accumulator.init(nu);

    std::vector<int> class_sizes = partition_sizes(nu);
    int k = class_sizes.size();
    int n_events = n - k;
    std::mt19937_64 rng(topology_stream_seed(seed, nu));
    genealogy_workspace w;

    int Muse = (n_events == 0) ? 1 : M;
    for (int j = 0; j < Muse; j++) {
      const double* ts = (n_events > 0) ? &times[k][(size_t) j * n_events] : NULL;
      accumulate_genealogy(accumulator, class_sizes, ts, tau, rng, w, check_identity);
    }

    // normalization
    for (size_t j = 0; j < accumulator.sizes.size(); j++) { // source
      int c = accumulator.sizes[j];
      for (int d = 1; d <= c; d++) {
        accumulator.esc_rate_ref(c, d) /= ((double) Muse * nu[c]);
        for (size_t j2 = 0; j2 < accumulator.sizes.size(); j2++) { // target
          int c2 = accumulator.sizes[j2];
          double pairs = (double) nu[c] * (nu[c2] - (c2 == c ? 1 : 0));

          if (pairs > 0) accumulator.move_rate_ref(c, c2, d) /= ((double) Muse * pairs);
          else           accumulator.move_rate_ref(c, c2, d) = 0.0;
        }
      }
    }
    ident[i] = accumulator.max_identity_error;
  }

  max_identity_error = 0.0;
  for (int i = 0; i < np; i++) {
    max_identity_error = std::max(max_identity_error, ident[i]);
  }
}

}

#endif