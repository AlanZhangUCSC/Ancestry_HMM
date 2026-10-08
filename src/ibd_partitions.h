#ifndef __IBD_PARTITIONS_H
#define __IBD_PARTITIONS_H

// State space... deal with partitions of the IBD classes and ancestry

#include "ibd_utils.h"
#include <unordered_map>

namespace ibd {

inline void enumerate_partitions_rec(int remaining, int maxpart, std::vector<int> &nu, std::vector<std::vector<int> > &out) {
    if (remaining == 0) {
        out.push_back(nu);
        return;
    }
    for (int c = std::min(remaining, maxpart); c >= 1; c --) {
        nu[c] ++;
        enumerate_partitions_rec(remaining - c, c, nu, out);
        nu[c] --;
    }
}

/// all partitions of k as multiplicity vectors of length n+1 (k <= n); k = 0 gives the empty partition
inline std::vector<std::vector<int> > enumerate_partitions(int k, int n) {
    std::vector<std::vector<int> > out;
    std::vector<int> nu(n + 1, 0);
    enumerate_partitions_rec(k, k, nu, out);
    return out;
}

inline int partition_parts(const std::vector<int> &nu) {
    int K = 0;
    for (size_t c = 1; c < nu.size(); c ++) K += nu[c];
    return K;
}

inline int partition_total(const std::vector<int> &nu) {
    int s = 0;
    for (size_t c = 1; c < nu.size(); c ++) s += (int) c * nu[c];
    return s;
}

/// class sizes in non-increasing order
inline std::vector<int> partition_sizes(const std::vector<int> &nu) {
    std::vector<int> sizes;
    for (int c = (int) nu.size() - 1; c >= 1; c --) {
        for (int i = 0; i < nu[c]; i ++) sizes.push_back(c);
    }
    return sizes;
}

/// human readable label: sizes joined by '.', e.g. "3.1.1"; "-" if empty
inline std::string partition_label(const std::vector<int> &nu) {
    std::vector<int> sizes = partition_sizes(nu);
    if (sizes.empty()) return "-";
    std::ostringstream os;
    for (size_t i = 0; i < sizes.size(); i ++) {
        if (i > 0) os << ".";
        os << sizes[i];
    }
    return os.str();
}

class colored_state {
public:
    std::vector<int> b0, b1;   /// multiplicities by class size (index 1..n) for ancestry 0 and 1
    int sub0, sub1;            /// index of b0 / b1 in state_space::subparts
    int part;                  /// index of the uncolored partition b0 + b1 in state_space::partitions
    int N0, N1;                /// chromosome copies of ancestry 0 / 1  (N_j(b) = sum_c c b_jc)
    int K0, K1;                /// number of IBD classes of ancestry 0 / 1
    std::string label;         /// "<ancestry-0 classes>|<ancestry-1 classes>", e.g. "2.1|1"
};

class state_space {
public:
    int n;
    int max_coal;
    size_t full_size;                                   /// size of the untruncated state space
    std::vector<std::vector<int> > partitions;          /// retained uncolored partitions of n
    std::unordered_map<std::vector<int>, int, vec_hash> partition_index;
    std::vector<std::vector<int> > subparts;            /// ancestry-specific multisets used by the states
    std::unordered_map<std::vector<int>, int, vec_hash> subpart_index;
    std::vector<colored_state> states;
    std::unordered_map<std::vector<int>, int, vec_hash> state_index;

    static std::vector<int> key(const std::vector<int> &b0, const std::vector<int> &b1) {
        std::vector<int> k(b0);
        k.insert(k.end(), b1.begin(), b1.end());
        return k;
    }

    /// index of a state, -1 if it is not part of the (possibly truncated) state space
    int find(const std::vector<int> &b0, const std::vector<int> &b1) const {
        std::unordered_map<std::vector<int>, int, vec_hash>::const_iterator it = state_index.find(key(b0, b1));
        if (it == state_index.end()) return -1;
        return it->second;
    }

    size_t size() const { return states.size(); }

    void build(int n_, int max_coal_);
};

/// number of colored states of the untruncated space: sum_k p(k) p(n-k)
inline size_t full_state_space_size(int n) {
    std::vector<double> p(n + 1, 0.0);
    p[0] = 1;
    for (int part = 1; part <= n; part ++) {
        for (int k = part; k <= n; k ++) p[k] += p[k-part];
    }
    double s = 0;
    for (int k = 0; k <= n; k ++) s += p[k] * p[n-k];
    return (size_t) s;
}

/// number of colored states retained when n - K <= max_coal
inline size_t truncated_state_space_size(int n, int max_coal) {
    std::vector<std::vector<int> > parts = enumerate_partitions(n, n);
    size_t total = 0;
    for (size_t p = 0; p < parts.size(); p ++) {
        if (n - partition_parts(parts[p]) > max_coal) continue;
        size_t prod = 1;
        for (int c = 1; c <= n; c ++) prod *= (size_t)(parts[p][c] + 1);
        total += prod;
    }
    return total;
}

inline bool colored_state_order(const colored_state &a, const colored_state &b) {
    if (a.N0 != b.N0) return a.N0 > b.N0;                 /// same column order as the original posterior files
    if (a.K0 + a.K1 != b.K0 + b.K1) return a.K0 + a.K1 > b.K0 + b.K1;   /// less IBD first
    return a.label < b.label;
}

inline void state_space::build(int n_, int max_coal_) {

    n = n_;
    max_coal = (max_coal_ < 0 || max_coal_ > n - 1) ? n - 1 : max_coal_;
    full_size = full_state_space_size(n);

    partitions.clear(); partition_index.clear();
    subparts.clear(); subpart_index.clear();
    states.clear(); state_index.clear();

    /// retained uncolored partitions
    std::vector<std::vector<int> > all = enumerate_partitions(n, n);
    for (size_t p = 0; p < all.size(); p ++) {
        if (n - partition_parts(all[p]) <= max_coal) {
            partition_index[ all[p] ] = (int) partitions.size();
            partitions.push_back(all[p]);
        }
    }

    /// all colorings of every retained partition: for each size c choose b0c in 0..nu_c
    std::vector<colored_state> tmp;
    for (size_t p = 0; p < partitions.size(); p ++) {
        const std::vector<int> &nu = partitions[p];
        std::vector<int> sizes;
        for (int c = 1; c <= n; c ++) if (nu[c] > 0) sizes.push_back(c);
        std::vector<int> choice(sizes.size(), 0);
        while (true) {
            colored_state s;
            s.b0.assign(n + 1, 0);
            s.b1.assign(n + 1, 0);
            for (size_t i = 0; i < sizes.size(); i ++) {
                s.b0[ sizes[i] ] = choice[i];
                s.b1[ sizes[i] ] = nu[ sizes[i] ] - choice[i];
            }
            s.part = (int) p;
            s.N0 = partition_total(s.b0);
            s.N1 = partition_total(s.b1);
            s.K0 = partition_parts(s.b0);
            s.K1 = partition_parts(s.b1);
            s.label = partition_label(s.b0) + "|" + partition_label(s.b1);
            tmp.push_back(s);

            /// next coloring (odometer)
            size_t i = 0;
            while (i < sizes.size()) {
                choice[i] ++;
                if (choice[i] <= nu[ sizes[i] ]) break;
                choice[i] = 0;
                i ++;
            }
            if (i == sizes.size()) break;
        }
    }

    std::sort(tmp.begin(), tmp.end(), colored_state_order);

    for (size_t s = 0; s < tmp.size(); s ++) {
        colored_state &st = tmp[s];
        if (subpart_index.find(st.b0) == subpart_index.end()) {
            subpart_index[ st.b0 ] = (int) subparts.size();
            subparts.push_back(st.b0);
        }
        if (subpart_index.find(st.b1) == subpart_index.end()) {
            subpart_index[ st.b1 ] = (int) subparts.size();
            subparts.push_back(st.b1);
        }
        st.sub0 = subpart_index[ st.b0 ];
        st.sub1 = subpart_index[ st.b1 ];
        state_index[ key(st.b0, st.b1) ] = (int) states.size();
        states.push_back(st);
    }
}

}

#endif
