#ifndef __IBD_DEATH_PROCESS_H
#define __IBD_DEATH_PROCESS_H

// death process of the truncated genealogy conditional on its IBD partition

#include "ibd_utils.h"

namespace ibd {

class death_process {
public:
    int n, K;
    double tau;
    double Lambda;                     /// uniformization rate (= lambda_n)
    int R;                             /// largest number of uniformized events tabulated
    std::vector<double> lam;           /// lam[a] = a(a-1)/2
    std::vector<double> loglam;        /// log(lam[a] / Lambda)
    std::vector<double> logstay;       /// log(1 - lam[a] / Lambda)
    std::vector<double> logw;          /// logw[ r*(n+1) + a ] = log [ (I+G/Lambda)^r ]_{a,K}
    std::vector<double> lfact;         /// lgamma(r+1)

    void init(int n_, int K_, double tau_);

    /// log P_{a->K}(s) and log P_{a-1->K}(s)
    void log_trans_pair(int a, double s, double &lp_a, double &lp_am1) const;
    double log_trans(int a, double s) const {
        double x, y;
        log_trans_pair(a, s, x, y);
        return x;
    }

    /// n-K coalescence times (increasing, in (0,tau)) from n-K uniforms
    void sample_times(const double *u, double *times) const;

private:
    inline double lw(int r, int a) const { return logw[ (size_t) r * (n + 1) + a ]; }
};

inline void death_process::init(int n_, int K_, double tau_) {
    n = n_; K = K_; tau = tau_;
    if (K < 1 || K > n) error("death_process: invalid number of lineages");
    lam.assign(n + 1, 0.0);
    for (int a = 0; a <= n; a ++) lam[a] = 0.5 * a * (a - 1.0);
    Lambda = (n >= 2) ? lam[n] : 1.0;
    double mu_max = Lambda * tau;
    R = (n - K) + (int) std::ceil(mu_max + 12.0 * std::sqrt(mu_max + 1.0) + 40.0);

    loglam.assign(n + 1, NEG_INF);
    logstay.assign(n + 1, 0.0);
    for (int a = 0; a <= n; a ++) {
        double p = lam[a] / Lambda;
        loglam[a] = (p > 0) ? std::log(p) : NEG_INF;
        logstay[a] = (p < 1) ? std::log1p(-p) : NEG_INF;
    }

    logw.assign((size_t)(R + 1) * (n + 1), NEG_INF);
    logw[ K ] = 0.0;
    for (int r = 1; r <= R; r ++) {
        size_t row = (size_t) r * (n + 1), prev = (size_t)(r - 1) * (n + 1);
        for (int a = K; a <= n; a ++) {
            double stay = (logw[prev+a] == NEG_INF || logstay[a] == NEG_INF) ? NEG_INF : logw[prev+a] + logstay[a];
            double move = NEG_INF;
            if (a - 1 >= K && logw[prev+a-1] != NEG_INF && loglam[a] != NEG_INF) move = logw[prev+a-1] + loglam[a];
            logw[row+a] = log_add(stay, move);
        }
    }

    lfact.resize(R + 1);
    for (int r = 0; r <= R; r ++) lfact[r] = std::lgamma(r + 1.0);
}

inline void death_process::log_trans_pair(int a, double s, double &lp_a, double &lp_am1) const {
    double mu = Lambda * s;
    if (s <= 0 || mu <= 0) {
        lp_a = (a == K) ? 0.0 : NEG_INF;
        lp_am1 = (a - 1 == K) ? 0.0 : NEG_INF;
        return;
    }
    double lmu = std::log(mu);
    int r0 = std::max(0, a - 1 - K);
    int r1 = std::min(R, (a - K) + (int) std::ceil(mu + 12.0 * std::sqrt(mu + 1.0) + 40.0));
    log_accumulator acc_a, acc_b;
    bool has_b = (a - 1 >= K);
    for (int r = r0; r <= r1; r ++) {
        double lp = -mu + (r > 0 ? r * lmu : 0.0) - lfact[r];
        double wa = lw(r, a);
        if (wa != NEG_INF) acc_a.add(lp + wa);
        if (has_b) {
            double wb = lw(r, a - 1);
            if (wb != NEG_INF) acc_b.add(lp + wb);
        }
    }
    lp_a = acc_a.value();
    lp_am1 = has_b ? acc_b.value() : NEG_INF;
}

inline void death_process::sample_times(const double *u, double *times) const {
    int a = n;
    double cur = 0.0;
    for (int i = 0; i < n - K; i ++) {
        double s = tau - cur;
        double ui = u[i];
        double rate = lam[a] - lam[K];
        double z = rate * s;
        /// single-remaining-event solution: truncated exponential with rate lam[a]-lam[K]
        double x_single = (z < 1e-12) ? ui * s : -std::log1p(ui * std::expm1(-z)) / rate;
        double x;
        if (a == K + 1) {
            x = x_single;
        }
        else {
            double lpa, dummy;
            log_trans_pair(a, s, lpa, dummy);
            double target = std::log1p(-ui) + lpa;
            double lo = 0.0, hi = s;
            x = std::min(std::max(x_single, 1e-3 * s), 0.999 * s);
            for (int it = 0; it < 200; it ++) {
                double lp, lpm1;
                log_trans_pair(a, s - x, lp, lpm1);
                double g = -lam[a] * x + lp - target;
                if (g > 0) lo = x; else hi = x;
                double dg = -lam[a] * std::exp(lpm1 - lp);
                double xn = x - g / dg;
                if (!(xn > lo && xn < hi) || !std::isfinite(xn)) xn = 0.5 * (lo + hi);
                if (std::fabs(xn - x) <= 1e-14 * s || hi - lo <= 1e-15 * s) {
                    x = xn;
                    break;
                }
                x = xn;
            }
        }
        if (!(x > 0)) x = 1e-300;
        if (x >= s) x = s * (1.0 - 1e-15);
        cur += x;
        times[i] = cur;
        a --;
    }
}

/// log P_{n->K}(tau) for all K = 1..n (index K), via one bridge per K
inline std::vector<double> log_lineage_distribution(int n, double tau) {
    std::vector<double> out(n + 1, NEG_INF);
    for (int K = 1; K <= n; K ++) {
        death_process b;
        b.init(n, K, tau);
        out[K] = b.log_trans(n, tau);
    }
    return out;
}

}

#endif
