/*
 Unit tests for the ancestry-IBD extension.

 Build (from src/):  make test_ibd      or
   g++ -O2 -std=c++11 -I. -o tests/test_ibd tests/test_ibd.cpp -larmadillo
*/

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <cstring>
using namespace std;

#define ARMA_NO_DEBUG
#include <armadillo>
using namespace arma;

/// original Ancestry_HMM pieces needed to compare emissions
#include "factorial.h"
#include "nchoosek.h"
#include "multichoose.h"
#include "multipermute.h"
#include "ancestry_pulse.h"
#include "input_line.h"
#include "pulses_to_ancestry.h"
#include "binomial.h"
#include "distribute_alleles.h"
#include "read_emissions.h"
#include "create_states.h"
#include "exponentiate_matrix.h"
#include "transition_information.h"
#include "create_transition_rates.h"

#include "ibd_hmm.h"

static int n_fail = 0, n_pass = 0;
#define CHECK(cond, msg) do {if (cond) {n_pass ++;} else {n_fail ++; cerr << "FAIL: " << msg << "  [" << #cond << "]  (" << __FILE__ << ":" << __LINE__ << ")\n";}} while (0)

static double rel(double a, double b) {return fabs(a - b) / max(1e-300, max(fabs(a), fabs(b)));}

/// --------------------------------------------------------------------------------------------
void test_state_space() {
    int expected[] = {0, 2, 5, 10, 20, 36, 65, 110, 185, 300, 481};
    for (int n = 1; n <= 10; n ++) {
        ibd::state_space sp;
        sp.build(n, -1);
        CHECK((int) sp.size() == expected[n], "state count n=" << n << " got " << sp.size());
        CHECK(sp.full_size == (size_t) expected[n], "full size formula n=" << n);
        /// every state is a valid configuration and is found by its key
        bool ok = true;
        for (size_t s = 0; s < sp.size(); s ++) {
            const ibd::colored_state &st = sp.states[s];
            if (st.N0 + st.N1 != n) ok = false;
            if (sp.find(st.b0, st.b1) != (int) s) ok = false;
       }
        CHECK(ok, "state consistency n=" << n);
   }
    ibd::state_space sp;
    sp.build(10, -1);
    CHECK(sp.partitions.size() == 42, "42 partitions of 10");
    ibd::state_space tr;
    tr.build(10, 4);
    CHECK(tr.size() == ibd::truncated_state_space_size(10, 4) && tr.size() == 210, "truncated size n=10, D=4: " << tr.size());
}

/// --------------------------------------------------------------------------------------------
void test_death_process() {
    /// transition probabilities vs dense matrix exponential of the death-process generator
    for (int n = 2; n <= 8; n ++) {
        mat G(n + 1, n + 1, fill::zeros);
        for (int a = 2; a <= n; a ++) {
            double l = 0.5 * a * (a - 1);
            G(a, a) = -l;
            G(a, a - 1) = l;
       }
        double taus[] = {0.001, 0.05, 0.3, 1.0, 3.0};
        for (int it = 0; it < 5; it ++) {
            double tau = taus[it];
            mat P = expmat(G * tau);
            double maxerr = 0;
            for (int K = 1; K <= n; K ++) {
                ibd::death_process b;
                b.init(n, K, tau);
                for (int a = K; a <= n; a ++) {
                    for (double frac = 0.25; frac <= 1.0; frac += 0.25) {
                        double s = tau * frac;
                        mat Ps = expmat(G * s);
                        double v = exp(b.log_trans(a, s));
                        if (Ps(a, K) > 1e-12) maxerr = max(maxerr, rel(v, Ps(a, K)));
                   }
               }
           }
            CHECK(maxerr < 1e-9, "death process P_{a->K} n=" << n << " tau=" << tau << " maxrel=" << maxerr);
       }
   }

    /// conditional coalescence times vs rejection sampling from the unconditioned process
    struct cfg {int n, K; double tau;} cases[] = {{4, 2, 0.5}, {3, 1, 1.0}, {5, 3, 0.2}, {6, 4, 0.1}};
    for (int c = 0; c < 4; c ++) {
        int n = cases[c].n, K = cases[c].K;
        double tau = cases[c].tau;
        int nev = n - K;
        ibd::death_process b;
        b.init(n, K, tau);
        std::mt19937_64 rng(1234 + c);
        int N = 200000;
        vector<double> m1(nev, 0.0), m2(nev, 0.0), u(nev), ts(nev);
        for (int s = 0; s < N; s ++) {
            for (int i = 0; i < nev; i ++) u[i] = ibd::open_uniform(rng);
            b.sample_times(&u[0], &ts[0]);
            for (int i = 0; i < nev; i ++) {m1[i] += ts[i] / N; m2[i] += ts[i] * ts[i] / N;}
       }
        /// rejection
        vector<double> r1(nev, 0.0);
        long acc = 0;
        std::exponential_distribution<double> ex(1.0);
        while (acc < N) {
            double cur = 0;
            vector<double> tt;
            int a = n;
            while (a > 1) {
                double l = 0.5 * a * (a - 1);
                cur += ex(rng) / l;
                if (cur > tau) break;
                tt.push_back(cur);
                a --;
           }
            if (a != K) continue;
            acc ++;
            for (int i = 0; i < nev; i ++) r1[i] += tt[i] / N;
       }
        bool ok = true;
        for (int i = 0; i < nev; i ++) {
            double sd = sqrt(max(0.0, m2[i] - m1[i] * m1[i]));
            double se = sd * sqrt(2.0 / N);
            if (fabs(m1[i] - r1[i]) > 5 * se) ok = false;
            cerr << "\t  times n=" << n << " K=" << K << " tau=" << tau << " event " << i << ": bridge mean " << m1[i] << " rejection mean " << r1[i] << " (se " << se << ")\n";
       }
        CHECK(ok, "conditional coalescence times agree with rejection sampling (n=" << n << ", K=" << K << ")");
   }
}

/// --------------------------------------------------------------------------------------------
/// brute force P(G = g | b): enumerate allele assignments to classes
vector<double> brute_allele_count(const ibd::colored_state &st, int n, const double *ref, bool fixed) {
    vector<pair<int,int> > classes;   /// (ancestry, size)
    for (int c = 1; c <= n; c ++) {
        for (int i = 0; i < st.b0[c]; i ++) classes.push_back(make_pair(0, c));
        for (int i = 0; i < st.b1[c]; i ++) classes.push_back(make_pair(1, c));
   }
    int K = (int) classes.size();
    int Kj[2] = {st.K0, st.K1};
    vector<double> pg(n + 1, 0.0);
    for (int mask = 0; mask < (1 << K); mask ++) {
        int q[2] = {0, 0};
        int g = 0;
        for (int h = 0; h < K; h ++) {
            if (mask & (1 << h)) {q[classes[h].first] ++; g += classes[h].second;}
       }
        double p = 1;
        for (int j = 0; j < 2; j ++) {
            double CA = ref[2 * j], Ca = ref[2 * j + 1];
            if (fixed) {
                double f = CA / (CA + Ca);
                p *= pow(f, q[j]) * pow(1 - f, Kj[j] - q[j]);
           }
            else {
                double a = CA + 1, b = Ca + 1;
                p *= exp(lgamma(a + q[j]) + lgamma(b + Kj[j] - q[j]) - lgamma(a + b + Kj[j]) - (lgamma(a) + lgamma(b) - lgamma(a + b)));
           }
       }
        pg[g] += p;
   }
    return pg;
}

void test_emissions() {
    std::mt19937_64 rng(7);
    for (int n = 1; n <= 7; n ++) {
        ibd::state_space sp;
        sp.build(n, -1);
        ibd::emission_model em;
        em.init(sp);
        for (int rep = 0; rep < 6; rep ++) {
            double ref[4] = {(double)(rng() % 40), (double)(rng() % 40), (double)(rng() % 40), (double)(rng() % 40)};
            bool fixed = (rep % 3 == 2);
            if (fixed) {ref[0] += 1; ref[1] += 1; ref[2] += 1; ref[3] += 1;}
            ibd::site_weights sw;
            ibd::compute_site_weights(em, ref, fixed, sw);
            /// W sums to one
            double maxw = 0;
            for (size_t i = 0; i < sw.W0.size(); i ++) {
                maxw = max(maxw, fabs(accumulate(sw.W0[i].begin(), sw.W0[i].end(), 0.0) - 1.0));
                maxw = max(maxw, fabs(accumulate(sw.W1[i].begin(), sw.W1[i].end(), 0.0) - 1.0));
           }
            CHECK(maxw < 1e-12, "W sums to one n=" << n);
            /// P(G=g|b) vs brute force
            double maxe = 0;
            for (size_t s = 0; s < sp.size(); s ++) {
                vector<double> a = ibd::allele_count_distribution(sp, sw, (int) s);
                vector<double> b = brute_allele_count(sp.states[s], n, ref, fixed);
                for (int g = 0; g <= n; g ++) maxe = max(maxe, fabs(a[g] - b[g]));
           }
            CHECK(maxe < 1e-12, "P(G=g|b) matches brute-force enumeration n=" << n << " err=" << maxe);
       }
   }

    /// all-singleton states vs original Ancestry_HMM emissions: ratio must be state independent
    for (int n = 1; n <= 4; n ++) {
        ibd::state_space sp;
        sp.build(n, -1);
        ibd::emission_model em;
        em.init(sp);
        vector<pulse> pulses(2);
        pulses[0].type = 0; pulses[1].type = 1;
        map<int, vector<vector<int> > > state_list;
        double nd = n;
        create_initial_states(nd, pulses, state_list);
        for (int rep = 0; rep < 8; rep ++) {
            bool fixed = (rep % 2 == 1);
            input_line line;
            line.reference_counts.resize(2);
            for (int j = 0; j < 2; j ++) {
                double A = 1 + rng() % 30, a = 1 + rng() % 30;
                line.reference_counts[j].push_back(A);
                line.reference_counts[j].push_back(a);
                line.reference_counts[j].push_back(A + a);
           }
            double rA = rng() % 6, ra = rng() % 6;
            if (rA + ra == 0) rA = 1;
            line.sample_counts.resize(1);
            line.sample_counts[0].push_back(rA);
            line.sample_counts[0].push_back(ra);
            line.sample_counts[0].push_back(rA + ra);
            line.error_1 = 0.01; line.error_2 = 0.01;
            vec orig;
            create_emissions_matrix(nd, line, fixed, state_list[n], 0, pulses, orig);

            double ref[4] = {line.reference_counts[0][0], line.reference_counts[0][1], line.reference_counts[1][0], line.reference_counts[1][1]};
            ibd::site_weights sw;
            ibd::compute_site_weights(em, ref, fixed, sw);
            vector<double> Lt;
            ibd::read_likelihood(rA, ra, 0.01, 0.01, n, Lt);
            vec e;
            vector<vector<double> > V;
            ibd::state_emissions(sp, em, sw, Lt, e, V);
            vector<double> ratios;
            for (size_t s = 0; s < state_list[n].size(); s ++) {
                int N0 = state_list[n][s][0], N1 = state_list[n][s][1];
                vector<int> b0(n + 1, 0), b1(n + 1, 0);
                b0[1] = N0; b1[1] = N1;
                int idx = sp.find(b0, b1);
                ratios.push_back(orig(s) / e(idx));
           }
            double mx = *max_element(ratios.begin(), ratios.end()), mn = *min_element(ratios.begin(), ratios.end());
            CHECK((mx - mn) / mx < 1e-9, "no-IBD emissions proportional to the original (n=" << n << ", fixed=" << fixed << ") spread " << (mx - mn) / mx);
       }
   }

    /// genotype likelihood: brute force over per-copy error patterns
    for (int n = 1; n <= 5; n ++) {
        for (int a = 0; a <= n; a ++) {
            vector<double> Lt;
            double e1 = 0.03, e2 = 0.07;
            double lmax = ibd::genotype_likelihood(a, n - a, e1, e2, n, Lt);
            double maxe = 0;
            for (int g = 0; g <= n; g ++) {
                /// copies 0..g-1 carry A; enumerate observed patterns
                double p = 0;
                for (int mask = 0; mask < (1 << n); mask ++) {
                    if (__builtin_popcount(mask) != a) continue;
                    double q = 1;
                    for (int c = 0; c < n; c ++) {
                        bool obsA = mask & (1 << c);
                        bool trueA = c < g;
                        if (trueA) q *= obsA ? 1 - e1 : e1;
                        else q *= obsA ? e2 : 1 - e2;
                   }
                    p += q;
               }
                maxe = max(maxe, fabs(Lt[g] * exp(lmax) - p));
           }
            CHECK(maxe < 1e-12, "genotype likelihood n=" << n << " a=" << a);
       }
   }
}

/// --------------------------------------------------------------------------------------------
void test_kernels() {
    /// n = 1 reduces to the original single-chromosome rates
    {
        ibd::state_space sp;
        sp.build(1, -1);
        double two_N = 2e4, t = 350, m = 0.3;
        ibd::rate_model rm;
        rm.build(sp, t, two_N, m, 10, 1, false, true);
        double l0 = two_N * m * (1 - exp(-t / two_N)), l1 = two_N * (1 - m) * (1 - exp(-t / two_N));
        int s0 = sp.find(vector<int>{0, 1}, vector<int>{0, 0}), s1 = sp.find(vector<int>{0, 0}, vector<int>{0, 1});
        CHECK(rel(rm.Q_raw(s0, s1), l0) < 1e-12, "n=1 rate 0->1 " << rm.Q_raw(s0, s1) << " vs " << l0);
        CHECK(rel(rm.Q_raw(s1, s0), l1) < 1e-12, "n=1 rate 1->0 " << rm.Q_raw(s1, s0) << " vs " << l1);
        /// and the original code's create_transition_rates for the default two-pulse model
        vector<pulse> pulses(2);
        pulses[0].type = 0; pulses[0].time = 100000; pulses[0].fraction_of_remainder = 1; pulses[0].entry_order = 0;
        pulses[1].type = 1; pulses[1].time = t; pulses[1].fraction_of_remainder = 1; pulses[1].entry_order = 1;
        vector<double> prop(2); prop[0] = 1 - m; prop[1] = m;
        mat R = create_transition_rates(pulses, two_N, prop);
        CHECK(rel(R(0, 1), l0) < 1e-12 && rel(R(1, 0), l1) < 1e-12, "original create_transition_rates agrees: " << R(0, 1) << " " << R(1, 0));
   }

    /// n = 2 analytic kernels
    {
        ibd::state_space sp;
        sp.build(2, -1);
        double tau = 0.4;
        vector<ibd::partition_kernels> ker;
        double ident;
        int M = 400000;
        ibd::compute_partition_kernels(sp, tau, M, 99, true, ker, ident);
        CHECK(ident < 1e-10, "branch-length identity n=2 err " << ident);
        for (size_t p = 0; p < sp.partitions.size(); p ++) {
            const vector<int> &nu = sp.partitions[p];
            if (nu[1] == 2) {
                double esc = (1 - exp(-2 * tau)) / 2;
                double mv = (tau - (1 - exp(-2 * tau)) / 2) / 2;
                CHECK(rel(ker[p].get_esc_rate(1, 1), esc) < 1e-12, "n=2 {1,1} escape");
                CHECK(rel(ker[p].get_move_rate(1, 1, 1), mv) < 1e-12, "n=2 {1,1} move");
           }
            else {
                double e21 = exp(-tau) / (1 - exp(-tau)) * (tau - (1 - exp(-2 * tau)) / 2);
                double e22 = 1 - tau * exp(-tau) / (1 - exp(-tau));
                cerr << "\t  n=2 {2}: esc(2,1) MC " << ker[p].get_esc_rate(2, 1) << " exact " << e21 << "; esc(2,2) MC " << ker[p].get_esc_rate(2, 2) << " exact " << e22 << "\n";
                CHECK(rel(ker[p].get_esc_rate(2, 1), e21) < 3e-3, "n=2 {2} escape d=1");
                CHECK(rel(ker[p].get_esc_rate(2, 2), e22) < 3e-3, "n=2 {2} escape d=2");
           }
       }
   }

    /// partitions {2,1,...,1}: one coalescence with density prop. to exp(-(n-1)x) on (0,tau);
    /// every kernel is a hand-derived function of T, integrated by Simpson's rule
    for (int n = 3; n <= 6; n ++) {
        ibd::state_space sp;
        sp.build(n, -1);
        double tau = 0.45;
        vector<ibd::partition_kernels> ker;
        double ident;
        ibd::compute_partition_kernels(sp, tau, 200000, 4242, false, ker, ident);
        vector<int> nu(n + 1, 0);
        nu[1] = n - 2; nu[2] = 1;
        int p = sp.partition_index[nu];
        const int NS = 20000;
        double ex[7] = {0, 0, 0, 0, 0, 0, 0}, ex2[7] = {0, 0, 0, 0, 0, 0, 0}, Z = 0;
        for (int i = 0; i <= NS; i ++) {
            double T = tau * i / NS;
            double wgt = (i == 0 || i == NS) ? 1 : (i % 2 ? 4 : 2);
            wgt *= exp(-(n - 1.0) * T);
            double A0 = n, A1 = n - 1, D0 = T, D1 = tau - T;
            double O0 = -expm1(-A0 * D0) / A0, O1 = -expm1(-A1 * D1) / A1;
            double P0 = ibd::phi_x(A0 * D0) / (A0 * A0), P1 = ibd::phi_x(A1 * D1) / (A1 * A1);
            double g1 = exp(-A1 * D1);
            double Y0 = O0 * g1, Y1 = O1;
            double v[7] = {
                Y0 + Y1,                        /// esc(1,1)
                2 * Y0,                         /// esc(2,1)
                Y1,                             /// esc(2,2)
                P0 + O0 * O1 + P1,              /// move(1->1,1)   (needs n >= 4)
                2 * P0 + O0 * O1 + P1,          /// move(1->2,1)
                2 * (P0 + O0 * O1),           /// move(2->1,1)
                P1                              /// move(2->1,2)
           };
            for (int k = 0; k < 7; k ++) {ex[k] += wgt * v[k]; ex2[k] += wgt * v[k] * v[k];}
            Z += wgt;
       }
        for (int k = 0; k < 7; k ++) {ex[k] /= Z; ex2[k] /= Z;}
        double mc[7] = {ker[p].get_esc_rate(1, 1), ker[p].get_esc_rate(2, 1), ker[p].get_esc_rate(2, 2), ker[p].get_move_rate(1, 1, 1),
                         ker[p].get_move_rate(1, 2, 1), ker[p].get_move_rate(2, 1, 1), ker[p].get_move_rate(2, 1, 2)};
        /// the Monte Carlo estimate is the sample mean of the kernel over M draws of T: compare in units of its standard error
        double worst = 0;
        for (int k = 0; k < 7; k ++) {
            if (k == 3 && n < 4) continue;
            double se = sqrt(max(0.0, ex2[k] - ex[k] * ex[k]) / 200000.0);
            worst = max(worst, fabs(mc[k] - ex[k]) / se);
       }
        cerr << "\t  n=" << n << " partition {2,1^" << n - 2 << "}: largest |MC - quadrature| = " << worst << " standard errors\n";
        CHECK(worst < 4.5, "kernels of partition {2,1^(n-2)} vs quadrature, n=" << n << " worst z " << worst);
   }

    /// generator properties for several n
    for (int n = 2; n <= 6; n ++) {
        ibd::state_space sp;
        sp.build(n, -1);
        double two_N = 2000, t = 300, m = 0.35;
        ibd::rate_model a;
        a.build(sp, t, two_N, m, 1000, 11, true, true);
        CHECK(a.identity_error < 1e-9, "identity n=" << n << " " << a.identity_error);
        CHECK(a.max_rowsum_error < 1e-12, "row sums n=" << n);
        double pisum = 0;
        for (size_t s = 0; s < sp.size(); s ++) pisum += exp(a.logpi[s]);
        CHECK(fabs(pisum - 1) < 1e-12 && fabs(a.retained_mass - 1) < 1e-10, "pi sums to one n=" << n << " " << a.retained_mass);
        /// symmetrized generator is exactly stationary
        vec pi(sp.size());
        for (size_t s = 0; s < sp.size(); s ++) pi(s) = exp(a.logpi[s]);
        rowvec r = pi.t() * a.Q;
        double scale = 0;
        for (size_t s = 0; s < sp.size(); s ++) scale += pi(s) * fabs(a.Q(s, s));
        CHECK(accu(abs(r)) / scale < 1e-12, "pi Q_rev = 0 n=" << n);

        /// the raw Monte Carlo generator is stationary and reversible up to Monte Carlo error:
        /// residuals (averaged over seeds) must shrink roughly like 1/sqrt(M)
        double st[2] = {0, 0}, db[2] = {0, 0};
        int Ms[2] = {500, 8000};
        for (int im = 0; im < 2; im ++) {
            for (int seed = 1; seed <= 4; seed ++) {
                ibd::rate_model b;
                b.build(sp, t, two_N, m, Ms[im], 1000 + 17 * seed, true, false);
                st[im] += b.stationarity_error / 4;
                db[im] += b.detailed_balance_asym / 4;
           }
       }
        cerr << "\t  n=" << n << " states=" << sp.size() << "  mean |piQ_raw| " << st[0] << " -> " << st[1]
             << "  mean detailed-balance asym " << db[0] << " -> " << db[1] << "  (M 500 -> 8000)\n";
        if (n > 2) {
            CHECK(st[1] < 0.6 * st[0], "raw stationarity residual shrinks with M (n=" << n << ")");
            CHECK(db[1] < 0.6 * db[0], "detailed-balance residual shrinks with M (n=" << n << ")");
       }
        CHECK(st[1] < 0.02 && db[1] < 0.02, "raw residuals small at M=8000 (n=" << n << ")");
   }
}

/// --------------------------------------------------------------------------------------------
void test_propagator() {
    for (int n = 2; n <= 6; n += 2) {
        ibd::state_space sp;
        sp.build(n, -1);
        ibd::rate_model rm;
        rm.build(sp, 500, 2e4, 0.4, 500, 5, true, false);
        ibd::propagator pe, pu;
        pe.build(rm.Q, rm.logpi); pe.force_method = 0;
        pu.build(rm.Q, rm.logpi); pu.force_method = 1;
        int S = (int) sp.size();
        std::mt19937_64 rng(3);
        double ds[] = {1e-6, 1e-4, 1e-3, 1e-2, 0.2};
        for (int k = 0; k < 5; k ++) {
            double d = ds[k];
            mat A = expmat(d * rm.Q);
            mat X(S, 3);
            for (int i = 0; i < S; i ++) for (int j = 0; j < 3; j ++) X(i, j) = ibd::open_uniform(rng);
            mat ref = A.t() * X;
            mat Xe = X, Xu = X;
            vector<int> meth;
            pe.forward(Xe, d, meth);
            pu.forward(Xu, d, meth);
            double ee = abs(Xe - ref).max() / abs(ref).max(), eu = abs(Xu - ref).max() / abs(ref).max();
            CHECK(ee < 1e-9 && eu < 1e-12, "forward propagation n=" << n << " d=" << d << " eigen " << ee << " unif " << eu);
            vec b = X.col(0);
            vec rb = A * b;
            vec be = b, bu = b;
            pe.backward(be, d, 0);
            pu.backward(bu, d, 1);
            double ebe = abs(be - rb).max() / abs(rb).max(), ebu = abs(bu - rb).max() / abs(rb).max();
            CHECK(ebe < 1e-9 && ebu < 1e-12, "backward propagation n=" << n << " d=" << d << " eigen " << ebe << " unif " << ebu);
            /// rows of exp(dQ) are probability distributions
            CHECK(abs(sum(A, 1) - 1).max() < 1e-10 && A.min() > -1e-12, "exp(dQ) stochastic n=" << n << " d=" << d);
       }
   }
}

/// --------------------------------------------------------------------------------------------
/// full forward algorithm against a dense reference implementation
void test_hmm_likelihood() {
    int n = 4;
    std::mt19937_64 rng(11);
    ibd::ibd_data data;
    int L = 300;
    data.counts.assign(2, vector<double>());
    data.block_start.push_back(0);
    for (int v = 0; v < L; v ++) {
        if (v == 180) data.block_start.push_back(v);
        data.chrom.push_back(v < 180 ? "c1" : "c2");
        data.pos.push_back(1000 * v);
        data.dist.push_back((v == 0 || v == 180) ? 0.0 : 1e-4 * (1 + (rng() % 50)));
        double f0 = ibd::open_uniform(rng), f1 = ibd::open_uniform(rng);
        data.ref.push_back(floor(40 * f0)); data.ref.push_back(40 - floor(40 * f0));
        data.ref.push_back(floor(40 * f1)); data.ref.push_back(40 - floor(40 * f1));
        data.err1.push_back(0.01); data.err2.push_back(0.01);
        for (int j = 0; j < 2; j ++) {
            int r = rng() % 7;
            int a = r ? rng() % (r + 1) : 0;
            data.counts[j].push_back(a);
            data.counts[j].push_back(r - a);
       }
   }
    data.block_start.push_back(L);

    ibd::ibd_settings set;
    set.two_N = 2000; set.m = 0.3; set.mc_samples = 300; set.verbose = false;
    vector<string> bases; bases.push_back("tests/tmp_a"); bases.push_back("tests/tmp_b");
    vector<int> ploidy(2, n);
    for (int method = -1; method <= 1; method ++) {
        set.force_method = method;
        ibd::ibd_model model;
        model.setup(data, bases, ploidy, set);
        model.set_time(150);
        double lnl = model.log_likelihood();

        /// reference: dense exp(dQ) and directly computed emissions
        ibd::ploidy_group &G = model.groups[0];
        int S = (int) G.sp.size();
        double ref_lnl = 0;
        for (int j = 0; j < 2; j ++) {
            vec alpha;
            for (int v = 0; v < L; v ++) {
                ibd::site_weights sw;
                ibd::compute_site_weights(G.em, &data.ref[4 * v], false, sw);
                vector<double> Lt;
                double lmax = ibd::read_likelihood(data.counts[j][2 * v], data.counts[j][2 * v + 1], 0.01, 0.01, n, Lt);
                vec e(S, fill::ones);
                if (lmax != ibd::NEG_INF) {
                    for (int s = 0; s < S; s ++) {
                        vector<double> pg = ibd::allele_count_distribution(G.sp, sw, s);
                        double val = 0;
                        for (int g = 0; g <= n; g ++) val += pg[g] * Lt[g];
                        e(s) = val;
                   }
               }
                else lmax = 0;
                if (v == 0 || v == 180) alpha = G.pi % e;
                else alpha = (expmat(data.dist[v] * G.rm.Q).t() * alpha) % e;
                double s = accu(alpha);
                alpha /= s;
                ref_lnl += log(s) + lmax;
           }
       }
        CHECK(fabs(lnl - ref_lnl) < 1e-7 * fabs(ref_lnl), "forward log-likelihood (method " << method << ") " << lnl << " vs dense " << ref_lnl);

        if (method == -1) {
            /// posterior decoding writes normalized marginals
            model.decode_posterior();
            ifstream in("tests/tmp_a.posterior");
            string line;
            getline(in, line);
            int rows = 0;
            double worst = 0;
            while (getline(in, line)) {
                istringstream ss(line);
                string c; int p;
                ss >> c >> p;
                double x, tot = 0;
                while (ss >> x) tot += x;
                worst = max(worst, fabs(tot - 1));
                rows ++;
           }
            CHECK(rows == L && worst < 1e-5, "posterior rows " << rows << " normalization error " << worst);

            /// posterior at a site equals brute-force smoothing with dense matrices (sample 0, block 1)
            vec post_ref;
            {
                int v_target = 90;
                vector<vec> E(180);
                for (int v = 0; v < 180; v ++) {
                    ibd::site_weights sw;
                    ibd::compute_site_weights(G.em, &data.ref[4 * v], false, sw);
                    vector<double> Lt;
                    double lmax = ibd::read_likelihood(data.counts[0][2 * v], data.counts[0][2 * v + 1], 0.01, 0.01, n, Lt);
                    E[v].ones(S);
                    if (lmax == ibd::NEG_INF) continue;
                    for (int s = 0; s < S; s ++) {
                        vector<double> pg = ibd::allele_count_distribution(G.sp, sw, s);
                        double val = 0;
                        for (int g = 0; g <= n; g ++) val += pg[g] * Lt[g];
                        E[v](s) = val;
                   }
               }
                vec a = G.pi % E[0]; a /= accu(a);
                for (int v = 1; v <= v_target; v ++) {a = (expmat(data.dist[v] * G.rm.Q).t() * a) % E[v]; a /= accu(a);}
                vec b(S, fill::ones);
                for (int v = 179; v > v_target; v --) {b = expmat(data.dist[v] * G.rm.Q) * (b % E[v]); b /= accu(b);}
                post_ref = a % b; post_ref /= accu(post_ref);
           }
            vec marg(n + 1, fill::zeros);
            for (int s = 0; s < S; s ++) marg(G.sp.states[s].N0) += post_ref(s);
            ifstream in2("tests/tmp_a.posterior");
            getline(in2, line);
            for (int r = 0; r <= 90; r ++) getline(in2, line);
            istringstream ss(line);
            string c; int p;
            ss >> c >> p;
            double worst2 = 0;
            for (int k = n; k >= 0; k --) {double x; ss >> x; worst2 = max(worst2, fabs(x - marg(k)));}
            CHECK(worst2 < 1e-5, "posterior marginal equals dense forward-backward (max diff " << worst2 << ")");
       }
   }
}

int main() {
    cerr.precision(8);
    cerr << "state space\n"; test_state_space();
    cerr << "death process\n"; test_death_process();
    cerr << "emissions\n"; test_emissions();
    cerr << "kernels and generator\n"; test_kernels();
    cerr << "propagation\n"; test_propagator();
    cerr << "HMM likelihood\n"; test_hmm_likelihood();
    cerr << "\n" << n_pass << " checks passed, " << n_fail << " failed\n";
    return n_fail ? 1 : 0;
}
