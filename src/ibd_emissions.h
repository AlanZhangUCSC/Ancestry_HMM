#ifndef __IBD_EMISSIONS_H
#define __IBD_EMISSIONS_H

// Emission probabilities of the ancestry-IBD HMM 

#include "ibd_utils.h"
#include "ibd_partitions.h"
#include <armadillo>

namespace ibd {

/// coefficients D(g, q) for one multiset of class sizes
class subpart_coef {
public:
    int N, K;
    std::vector<double> coef;    /// coef[q*(N+1) + g]
    void init(const std::vector<int> &nu) {
        N = partition_total(nu);
        K = partition_parts(nu);
        std::vector<double> poly((size_t)(K + 1) * (N + 1), 0.0);
        poly[0] = 1.0;
        int q_used = 0, g_used = 0;
        for (int c = 1; c < (int) nu.size(); c ++) {
            for (int r = 0; r < nu[c]; r ++) {
                /// multiply by (1 + u y^c)
                for (int q = q_used; q >= 0; q --) {
                    for (int g = g_used; g >= 0; g --) {
                        double v = poly[(size_t) q * (N + 1) + g];
                        if (v != 0) poly[(size_t)(q + 1) * (N + 1) + g + c] += v;
                    }
                }
                q_used += 1;
                g_used += c;
            }
        }
        coef.swap(poly);
    }
};

class emission_model {
public:
    int n;
    std::vector<subpart_coef> D;          /// one per state_space::subparts entry
    std::vector<int> sub0_list;           /// subparts used as ancestry-0 part by some state
    void init(const state_space &sp) {
        n = sp.n;
        D.resize(sp.subparts.size());
        for (size_t i = 0; i < sp.subparts.size(); i ++) D[i].init(sp.subparts[i]);
        std::vector<int> used(sp.subparts.size(), 0);
        for (size_t s = 0; s < sp.size(); s ++) used[sp.states[s].sub0] = 1;
        sub0_list.clear();
        for (size_t i = 0; i < used.size(); i ++) if (used[i]) sub0_list.push_back((int) i);
    }
};

/// R(q, K) for 0 <= q <= K <= n, stored at [K*(n+1) + q]
inline void beta_ratio_table(double CA, double Ca, bool fixed, int n, std::vector<double> &R) {
    R.assign((size_t)(n + 1) * (n + 1), 0.0);
    double C = CA + Ca;
    if (fixed && C > 0) {
        double f = CA / C;
        for (int K = 0; K <= n; K ++) {
            for (int q = 0; q <= K; q ++) {
                double lv = 0;
                if (q > 0) lv += (f > 0) ? q * std::log(f) : NEG_INF;
                if (K - q > 0) lv += (f < 1) ? (K - q) * std::log1p(-f) : NEG_INF;
                R[(size_t) K * (n + 1) + q] = (lv == NEG_INF) ? 0.0 : std::exp(lv);
            }
        }
        return;
    }
    double alpha = CA + 1.0, beta = Ca + 1.0;
    std::vector<double> la(n + 1, 0.0), lb(n + 1, 0.0), lab(n + 1, 0.0);
    for (int i = 1; i <= n; i ++) {
        la[i] = la[i-1] + std::log(alpha + i - 1);
        lb[i] = lb[i-1] + std::log(beta + i - 1);
        lab[i] = lab[i-1] + std::log(alpha + beta + i - 1);
    }
    for (int K = 0; K <= n; K ++) {
        for (int q = 0; q <= K; q ++) {
            R[(size_t) K * (n + 1) + q] = std::exp(la[q] + lb[K-q] - lab[K]);
        }
    }
}

/// W[sub][g] for every subpart, given the tables R for one ancestry
inline void subpart_weights(const emission_model &em, const std::vector<double> &R, std::vector<std::vector<double> > &W) {
    const int n = em.n;
    W.resize(em.D.size());
    for (size_t i = 0; i < em.D.size(); i ++) {
        const subpart_coef &dc = em.D[i];
        W[i].assign(dc.N + 1, 0.0);
        for (int q = 0; q <= dc.K; q ++) {
            double r = R[(size_t) dc.K * (n + 1) + q];
            if (r == 0) continue;
            const double *row = &dc.coef[(size_t) q * (dc.N + 1)];
            for (int g = 0; g <= dc.N; g ++) {
                if (row[g] != 0) W[i][g] += row[g] * r;
            }
        }
    }
}

/// per-site, per-ploidy quantities shared by all samples of that ploidy
class site_weights {
public:
    std::vector<std::vector<double> > W0, W1;
};

inline void compute_site_weights(const emission_model &em, const double *ref, bool fixed, site_weights &sw) {
    std::vector<double> R0, R1;
    beta_ratio_table(ref[0], ref[1], fixed, em.n, R0);
    beta_ratio_table(ref[2], ref[3], fixed, em.n, R1);
    subpart_weights(em, R0, sw.W0);
    subpart_weights(em, R1, sw.W1);
}

/// scaled read likelihood Lt(g) = L(g)/max_g L(g); returns log max_g L(g) (NEG_INF if no information)
inline double read_likelihood(double rA, double ra, double e1, double e2, int n, std::vector<double> &Lt) {
    Lt.assign(n + 1, 1.0);
    double r = rA + ra;
    if (r <= 0) return NEG_INF;
    std::vector<double> ll(n + 1);
    double lc = log_choose(r, rA);
    double mx = NEG_INF;
    for (int g = 0; g <= n; g ++) {
        double fr = (double) g / n;
        double p = fr * (1.0 - e1) + (1.0 - fr) * e2;
        double v = lc;
        if (rA > 0) v += (p > 0) ? rA * std::log(p) : NEG_INF;
        if (ra > 0 && v != NEG_INF) v += (p < 1) ? ra * std::log1p(-p) : NEG_INF;
        ll[g] = v;
        if (v > mx) mx = v;
    }
    if (mx == NEG_INF) return NEG_INF;
    for (int g = 0; g <= n; g ++) Lt[g] = (ll[g] == NEG_INF) ? 0.0 : std::exp(ll[g] - mx);
    return mx;
}

/// genotype likelihood: observed A count gA among n copies; per-copy errors e1 (A->a) and e2 (a->A)
inline double genotype_likelihood(double gA, double ga, double e1, double e2, int n, std::vector<double> &Lt) {
    Lt.assign(n + 1, 1.0);
    if (gA + ga <= 0) return NEG_INF;
    if ((int) std::floor(gA + ga + 0.5) != n) return NEG_INF;     /// inconsistent with the ploidy: treated as missing
    int a = (int) std::floor(gA + 0.5);
    std::vector<double> ll(n + 1, NEG_INF);
    double mx = NEG_INF;
    for (int g = 0; g <= n; g ++) {
        log_accumulator acc;
        for (int x = std::max(0, a - (n - g)); x <= std::min(a, g); x ++) {
            /// x observed A among the g true A copies, a - x observed A among the n - g true a copies
            double v = log_choose(g, x) + log_choose(n - g, a - x);
            if (x > 0) v += (e1 < 1) ? x * std::log1p(-e1) : NEG_INF;
            if (g - x > 0) v += (e1 > 0) ? (g - x) * std::log(e1) : NEG_INF;
            if (a - x > 0) v += (e2 > 0) ? (a - x) * std::log(e2) : NEG_INF;
            if (n - g - a + x > 0) v += (e2 < 1) ? (n - g - a + x) * std::log1p(-e2) : NEG_INF;
            acc.add(v);
        }
        ll[g] = acc.value();
        if (ll[g] > mx) mx = ll[g];
    }
    if (mx == NEG_INF) return NEG_INF;
    for (int g = 0; g <= n; g ++) Lt[g] = (ll[g] == NEG_INF) ? 0.0 : std::exp(ll[g] - mx);
    return mx;
}

/// emission vector over all states from site weights and the scaled likelihood Lt(g)
inline void state_emissions(const state_space &sp, const emission_model &em, const site_weights &sw,
                             const std::vector<double> &Lt, arma::vec &e, std::vector<std::vector<double> > &V) {
    const int n = sp.n;
    /// V[sub0][g1] = sum_g0 W0[sub0][g0] Lt(g0 + g1)
    V.resize(em.D.size());
    for (size_t ii = 0; ii < em.sub0_list.size(); ii ++) {
        int i = em.sub0_list[ii];
        int N0 = em.D[i].N;
        const std::vector<double> &w0 = sw.W0[i];
        V[i].assign(n - N0 + 1, 0.0);
        for (int g1 = 0; g1 <= n - N0; g1 ++) {
            double v = 0;
            for (int g0 = 0; g0 <= N0; g0 ++) v += w0[g0] * Lt[g0 + g1];
            V[i][g1] = v;
        }
    }
    e.set_size(sp.size());
    double mx = 0;
    for (size_t s = 0; s < sp.size(); s ++) {
        const colored_state &st = sp.states[s];
        const std::vector<double> &v = V[st.sub0];
        const std::vector<double> &w1 = sw.W1[st.sub1];
        double val = 0;
        for (int g1 = 0; g1 <= st.N1; g1 ++) val += v[g1] * w1[g1];
        e(s) = val;
        if (val > mx) mx = val;
    }
    if (!(mx > 0) || !std::isfinite(mx)) {
        e.ones();            /// bad site: no usable information
        return;
    }
    for (size_t s = 0; s < sp.size(); s ++) {
        if (e(s) < mx * 1e-300) e(s) = mx * 1e-300;
    }
}

/// P(G = g | b) for one state (used by tests)
inline std::vector<double> allele_count_distribution(const state_space &sp, const site_weights &sw, int s) {
    const colored_state &st = sp.states[s];
    std::vector<double> pg(sp.n + 1, 0.0);
    const std::vector<double> &w0 = sw.W0[st.sub0], &w1 = sw.W1[st.sub1];
    for (int g0 = 0; g0 <= st.N0; g0 ++) {
        for (int g1 = 0; g1 <= st.N1; g1 ++) pg[g0 + g1] += w0[g0] * w1[g1];
    }
    return pg;
}

}

#endif
