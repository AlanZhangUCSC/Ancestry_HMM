#ifndef __IBD_MAIN_H
#define __IBD_MAIN_H

#include "ibd_utils.h"
#include "ibd_hmm.h"

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


  std::cerr << "total run time:\t\t\t" << (double) (clock() - total) << " ms" << endl;
  return 0;
}

#endif