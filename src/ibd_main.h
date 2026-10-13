#ifndef __IBD_MAIN_H
#define __IBD_MAIN_H

#include "ibd_utils.h"
#include "ibd_hmm.h"
#include "ibd_golden_search.h"

int run_ancestry_ibd(cmd_line &options) {
  std::cout << "Running ancestry-IBD model" << std::endl;

  // time tracking
  clock_t t = clock();
  clock_t total = clock();
  options.viterbi = false; // Force Viterbi decoding to be off for IBD model for now.

  // chain objects for each sample
  std::vector<markov_chain> markov_chain_information;
  
  // Get sample ids and ploidy from sample file.. Using the same function as the ancestry HMM main.cpp.
  std::cerr << "\t\t\t\t" << (double) (clock() - t) << " ms\n" << "reading sample ids and ploidy from " << options.sample_file; t = clock();
  read_samples(markov_chain_information, options.sample_file, options.viterbi);

  // Read data from input file.
  std::cerr << "\t\t\t\t" << (double) (clock() - t) << " ms\n" << "reading data from " << options.input_file; t = clock();
  ibd::ibd_data data;
  ibd::read_ibd_input(options.input_file, 2, markov_chain_information.size(), options.minimum_distance, options.error_rates, options.error_rate, data);

  // Keep a separate ibd settings object for the IBD model.
  ibd::ibd_settings set(options);
  std::cerr << "\t[ibd] 2N = " << set.two_N << ", m = " << options.ancestry_proportion[1] << ", Monte Carlo genealogies per partition = " << set.mc_samples << ", seed = " << set.seed << "\n";

  // Initialize the IBD model.
  std::cerr << "\t\t\t\t" << (double) (clock() - t) << " ms\n" << "initializing IBD model"; t = clock();
  ibd::ibd_model model;
  model.setup(data, set, markov_chain_information);

  // Estimate admixture time
  int adm = (options.ancestry_pulses[0].time < options.ancestry_pulses[1].time) ? 0 : 1;
  bool estimate_t = !options.ancestry_pulses[adm].time_fixed;
  double t0 = options.ancestry_pulses[adm].time;
  double t_hat = t0;
  double lnl = 0;
  if (estimate_t) {
    std::cerr << "\t\t\t\t" << (double) (clock() - t) << " ms\n" << "estimating admixture time"; t = clock();
    std::cerr << "\t[ibd] golden-section search for t on log scale in [" << options.t_min << ", " << options.t_max << "]\n";
    t_hat = ibd::golden_section(model, options.t_min, options.t_max, options.tolerance, options.ibd_t_tol, lnl);
  }
  model.set_time(t_hat);
  lnl = model.log_likelihood();
  std::cerr << "\t[ibd] t = " << t_hat << "\tlnL = " << std::setprecision(12) << lnl << std::setprecision(6) << "\n";

  // Output the results. Using the same layout as the ancestry HMM main.cpp.
  std::cerr << "\ttype\ttime\tproportion\n";
  std::cout << "optimum: \n";
  std::cout << "\ttype\ttime\tproportion\n";
  for (int p = 0; p < 2; p ++) {
    int ty = (int) options.ancestry_pulses[p].type;
    double time = (p == adm) ? t_hat : options.ancestry_pulses[p].time;
    std::cerr << "\t" << ty << "\t" << time << "\t" << options.ancestry_proportion[ty] << "\n";
    std::cout << "\t" << ty << "\t" << time << "\t" << options.ancestry_proportion[ty] << "\n";
  }
  std::cout << "lnL:\t" << std::setprecision(12) << lnl << std::setprecision(6) << "\n";

  for (const auto &group : model.groups) {
    const ibd::rate_model &rm = group.rm;
    std::cerr << "\t[ibd] ploidy " << group.ploidy << ": retained stationary mass " << rm.retained_mass
              << ", raw Monte Carlo residuals |piQ| " << rm.stationarity_error
              << ", detailed-balance asymmetry " << rm.detailed_balance_asym << "\n";
  }

  if (set.write_q != "") {
    model.write_rate_matrices(set.write_q);
    std::cerr << "\t[ibd] rate matrices written to " << set.write_q << "\n";
  }

  // decode posterior probs
  std::cerr << "\t\t\t\t" << (double) (clock() - t) << " ms\n" << "decoding posterior probs"; t = clock();
  model.decode_posterior();
  if (model.n_bad_sites > 0) {
    std::cerr << "\t[ibd] warning: " << model.n_bad_sites << " site(s) had inconsistent genotype totals or numerical resets\n";
  }
  long ne = 0;
  long nu = 0;
  long nf = 0;
  for (const auto& group : model.groups) {
    ne += group.prop.n_eigen;
    nu += group.prop.n_unif;
    nf += group.prop.n_fallback;
  }
  std::cerr << "\t[ibd] propagation steps: eigen " << ne << ", uniformization " << nu << ", eigen->uniformization fall-backs " << nf << "\n" ;


  std::cerr << "total run time:\t\t\t" << (double) (clock() - total) << " ms" << endl;
  return 0;
}

#endif