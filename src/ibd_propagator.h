#ifndef __IBD_PROPAGATOR_H
#define __IBD_PROPAGATOR_H

// Propagate state probabilities over continuous time using a CTMC rate matrix.

#include "ibd_utils.h"
#include <armadillo>

namespace ibd {

class propagator {
public:
  int n_states;
  const arma::mat* Qp;        // rate matrix
  std::vector<double> logpi;    // log of pi (stationary distribution)
  arma::vec sqrtpi;           // square root of pi
  arma::vec isqrtpi;          // inverse of square root of pi
  double sqrtpi_max;          // max value of sqrtpi

  arma::mat U;                // Matrix of eigenvectors
  arma::vec lambda;           // Eigenvalues
  double Lambda;              // uniformization rate
  arma::sp_mat P, Pt;         // P = I + Q/Lambda and its transpose. **P is not the probability matrix**
  double nnz;                 // number of non-zero elements in P
  int force_method;           // -1 automatic, 0 eigen only (no fall-back), 1 uniformization only
  double risk_tol;
  long n_eigen;
  long n_unif;
  long n_fallback;
  bool eigen_allowed;         // requires a reversible Q
  double amortize_steps;      // steps over which the one-off O(S^3) eigendecomposition is amortized

  propagator() : n_states(0), sqrtpi_max(0), Lambda(0), nnz(0), force_method(-1), risk_tol(1e-10), n_eigen(0), n_unif(0), n_fallback(0), eigen_allowed(false), Qp(NULL), amortize_steps(1e4) {}

  void build(const arma::mat& _Q, const std::vector<double>& _logpi, double allow_eigen = true) {
    // initialize class members
    Qp = &_Q;
    logpi = _logpi;
    eigen_allowed = allow_eigen;
    U.reset();
    lambda.reset();
    n_states = (int) Q.n_rows;
    sqrtpi.set_size(n_states);
    isqrtpi.set_size(n_states);

    for (int i = 0; i < n_states; i++) {
      sqrtpi(i) = std::exp( 0.5 * logpi[i]);
      double v  = std::exp(-0.5 * logpi[i]);
      if (!std::isfinite(v) || sqrtpi(i) == 0) {
        v = 0.0;
        sqrtpi(i) = 0.0;
      }
      isqrtpi(i) = v;
    }
    sqrtpi_max = sqrtpi.max();

    // construct the uniformization matrix P = I + Q/Lambda
    Lambda = 0;
    for (int i = 0; i < n_states; i++) {
      Lambda = std::max(Lambda, -Q(i,i));
    }
    Lambda = (Lambda > 0) ? Lambda * (1.0 + 1e-12) : 1.0;
    arma::mat Pd = Q / Lambda;
    Pd.diag() += 1.0;
    Pd.elem(arma::find(Pd < 0)).zeros();
    P = arma::sp_mat(Pd);
    Pt = P.t();
    nnz = (double) P.n_nonzero;
  }

  void get_eigen() {
    if (U.n_elem > 0) return; // already computed

    arma::mat Sm(n_states, n_states, arma::fill::zeros); // symmetric_rate_matrix
    for (int i = 0; i < n_states; i++) {
      for (int j = 0; j < n_states; j++) {
        double q = (*Qp)(i,j);
        if (q == 0) continue;
        if (i == j) {
          Sm(i, j) = q;
          continue;
        }
        if (q > 0 && lpi[i] != NEG_INF && lpi[j] != NEGINF) {
          Sm(i, j) = std::exp(std::log(q) + 0.5 * (lpi[i] - lpi[j]));
        }
      }
    }
    
    Sm = 0.5 * (Sm + Sm.t());
    if (!arma::eig_sym(lambda, U, Sm)) {
      error("Eigendecomposition of the rate matrix failed :(");
    }
    lambda.elem(arma::find(lambda > 0)).zeros();
  }

  // force uniformation upper bound. A heuristic, not exact calculation.
  int unif_upper_bound(double mu) {
      if (mu <= 0) return 0;
      return (int) std::ceil(mu + 8.5 * std::sqrt(mu) + 12.0);
  }

  // estimate the cost of uniformization. Heuristic again.
  double unif_cost(double mu) {
    if (mu <= 0)  return 0;
    if (mu <= 50) return mu + 6.0 * std::sqrt(mu) + 8.0;
    return unif_upper_bound(mu);
  }

  // select the cheaper method between eigen and uniformization.
  int preferred_method(double d) const {
    if (!eigen_allowed)    return 1;
    if (force_method >= 0) return force_method;

    double cost_unif  = (unif_cost(Lambda * d) + 1.0) * 2.0 * nnz;
    double cost_eigen = 4.0 * (double) n_states * (double) n_states;

    if (U.n_elem == 0) cost_eigen += 10.0 * (double) S * S * S / std::max(1.0, amortize_steps);

    return (cost_unif <= cost_eigen) ? 1 : 0;
  }

  // propogate using uniformization.
  void unif_prop(const arma::sp_mat& Mat, arma::mat& X, double mu) const {
    if (mu <= 0) return;
    arma::mat V = X; // current matrix

    // small mu case
    if (mu <= 50) {
      double w = std::exp(-mu); // weight
      arma::mat R = w * V;      // result matrix
      for (int k = 1 ; ; k++) {
        V = Mat * V;
        w *= mu / k;
        R += w * V;
        if (k > mu && w * (k + 1.0) / (k + 1.0 - mu) < 1e-16) break;
      }
      X = R;
      return;
    }

    // large mu case
    int k_hi = unif_upper_bound(mu);
    int k_lo = std::max(0, (int) std::floor(mu - 8.5 * std::sqrt(mu) - 12.0));
    double lmu = std::log(mu); // compute in log space to avoid underflow

    arma::mat R(X.n_rows, X.n_cols, arma::fill::zeros);
    for (int k = 0; k <= k_hi; k++) {
      if (k >= k_lo) {
        double w = std::exp(-mu + k * lmu - std::lgamma(k + 1.0));
        if (w > 0) R += w * V;
      }
      if (k < k_hi) V = Mat * V;
    }
    X = R;
  }

  void clean_nonneg(arma::mat& X) {
    // set negative elements due to floating point errors to zero
    X.elem(arma::find(X < 0)).zeros();
  }

  // propagate state probs forward
  void forward(arma::mat& X, double d, std::vector<int>& method) {
    method.assign(X.n_cols, 0);
  }


};

}

#endif