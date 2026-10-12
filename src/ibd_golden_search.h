#ifndef __IBD_OPTIMIZE_H
#define __IBD_OPTIMIZE_H

/*
 Golden-section search for the admixture time t of the ancestry-IBD model.

 The search is carried out on log(t) between t_min and t_max: t is a scale
 parameter whose plausible range spans orders of magnitude.  Because the Monte
 Carlo rate kernels use common random numbers (same topologies and uniforms
 for every t, times obtained by inversion), lnL(t) is a smooth function and
 the bracketing search behaves as in the deterministic case.
*/

#include "ibd_hmm.h"
#include <functional>
#include <map>
#include <iomanip>

namespace ibd {

inline double golden_section(ibd_model &model, double t_min, double t_max, double lnl_tol, double rel_tol, double &best_lnl) {
    std::map<double, double> cache;
    std::function<double(double)> f = [&](double logt) {
        std::map<double, double>::iterator it = cache.find(logt);
        if (it != cache.end()) return it->second;
        model.set_time(std::exp(logt));
        double v = model.log_likelihood();
        cache[logt] = v;
        return v;
    };

    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;
    double a = std::log(t_min), b = std::log(t_max);
    double x1 = b - phi * (b - a), x2 = a + phi * (b - a);
    double f1 = f(x1), f2 = f(x2);

    std::cerr << "\titeration\tlow bound\tlow test\thigh test\thigh bound\tlnl low\tlnl high\n";
    for (int it = 0; it < 200; it ++) {
        std::cerr << "\t" << it << "\t" << std::exp(a) << "\t" << std::exp(x1) << "\t" << std::exp(x2) << "\t" << std::exp(b)
                  << "\t" << std::setprecision(12) << f1 << "\t" << f2 << std::setprecision(6) << "\n";
        double width = b - a;
        if (width < rel_tol) break;
        if (std::fabs(f1 - f2) < lnl_tol && width < 0.05) break;
        if (f2 >= f1) {
            a = x1; x1 = x2; f1 = f2;
            x2 = a + phi * (b - a);
            f2 = f(x2);
        }
        else {
            b = x2; x2 = x1; f2 = f1;
            x1 = b - phi * (b - a);
            f1 = f(x1);
        }
    }
    double best = 0.5 * (a + b);
    best_lnl = f(best);
    /// guard against an end-point optimum
    if (f1 > best_lnl) { best = x1; best_lnl = f1; }
    if (f2 > best_lnl) { best = x2; best_lnl = f2; }
    return std::exp(best);
}

}

#endif
