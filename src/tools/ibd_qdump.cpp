/*
 ibd_qdump: write the ancestry-IBD state space, stationary distribution,
 Monte Carlo kernels and rate matrix for given (n, t, 2N, m).

 usage: ibd_qdump n t two_N m [mc_samples=2000] [seed=20260101] [max_coal=-1] [--kernels] [--no-symmetrize]

 Output (stdout): the format of rate_model::write (STATE / RATE lines), and with
 --kernels one line per kernel:
   KMOVE <partition label> c c2 d value      (per ordered pair of distinct classes, scaled time)
   KESC  <partition label> c d value         (per class, scaled time)
 Used by tools/validate_ibd_msprime.py and tools/check_kernels_bruteforce.py.
*/

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cstring>
#include <algorithm>
using namespace std;
#define ARMA_NO_DEBUG
#include <armadillo>
using namespace arma;
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

int main(int argc, char *argv[]) {
    if (argc < 5) {
        cerr << "usage: ibd_qdump n t two_N m [mc_samples] [seed] [max_coal] [--kernels] [--no-symmetrize]\n";
        return 1;
    }
    int n = atoi(argv[1]);
    double t = atof(argv[2]), two_N = atof(argv[3]), m = atof(argv[4]);
    int M = 2000;
    unsigned long long seed = 20260101ULL;
    int max_coal = -1;
    bool kernels = false, sym = true;
    int pos = 0;
    for (int i = 5; i < argc; i ++) {
        if (strcmp(argv[i], "--kernels") == 0) { kernels = true; continue; }
        if (strcmp(argv[i], "--no-symmetrize") == 0) { sym = false; continue; }
        if (pos == 0) M = atoi(argv[i]);
        else if (pos == 1) seed = strtoull(argv[i], NULL, 10);
        else if (pos == 2) max_coal = atoi(argv[i]);
        pos ++;
    }
    ibd::state_space sp;
    sp.build(n, max_coal);
    ibd::rate_model rm;
    rm.build(sp, t, two_N, m, M, seed, sym, true);
    cout.precision(12);
    rm.write(sp, cout);
    if (kernels) {
        for (size_t p = 0; p < sp.partitions.size(); p ++) {
            const ibd::partition_kernels &k = rm.kernels[p];
            string lab = ibd::partition_label(sp.partitions[p]);
            for (size_t i = 0; i < k.sizes.size(); i ++) {
                int c = k.sizes[i];
                for (int d = 1; d <= c; d ++) {
                    cout << "KESC\t" << lab << "\t" << c << "\t" << d << "\t" << k.get_esc_rate(c, d) << "\n";
                    for (size_t i2 = 0; i2 < k.sizes.size(); i2 ++) {
                        int c2 = k.sizes[i2];
                        if (c2 == c && sp.partitions[p][c] < 2) continue;
                        cout << "KMOVE\t" << lab << "\t" << c << "\t" << c2 << "\t" << d << "\t" << k.get_move_rate(c, c2, d) << "\n";
                    }
                }
            }
        }
    }
    cerr << "identity error " << rm.identity_error << ", raw |piQ| " << rm.stationarity_error << ", raw detailed-balance asymmetry " << rm.detailed_balance_asym << "\n";
    return 0;
}
