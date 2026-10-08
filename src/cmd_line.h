#ifndef __CMD_LINE_H
#define __CMD_LINE_H

/// command line information and global parameters
class cmd_line {
public:
    
    /// terms to bound the optimization
    double t_max;
    double t_min;
    
    /// to bound proportion search
    double p_max;
    double p_min;
    
    /// to create intial simplex points
    double t_length;
    double p_length;
    
    /// number of restarts
    int n_restarts;
    
    /// proportion ancestry for 0-n must sum to 1
    /// these are therefore the final ancestry proportion, not necessary the proportion that fluxed if additional pulses occured closer to the present
    vector<double> ancestry_proportion;
    
    /// store relevant ancestry information
    vector<pulse> ancestry_pulses;
    
    /// diploid effective population size ( i.e. 2n )
    double ne;
    
    /// tolerance for parameter search
    double tolerance;
    
    /// minimum recombinational distance between markers
    double minimum_distance;
    
    /// error rates for reads (if read based) or genotypes (if genotype based)
    double error_rate;
    
    /// bool sample is expressed as genotypes, not read counts
    bool genotype;
    
    /// ancestral genotype frequencies are fixed
    bool ancestral_fixed;
    
    /// viterbi output
    /// caution: not recommended for more samples of ploidy > 1
    bool viterbi;
    
    /// number of digits of precision to include
    int precision;
    
    /// output actual pulses rather than ancestry states
    bool output_pulses;
    
    /// error rates specifed
    bool error_rates; 
    
    /// input file name
    string input_file;
    
    /// sample file
    string sample_file;
    
    /// bootstrap
    int n_bootstraps;
    int block_size; 

    /// IBD relevent parameters
    bool ibd;                      /// use the ancestry-IBD HMM
    int ibd_mc;                    /// Monte Carlo genealogies per IBD partition
    unsigned long long ibd_seed;   /// seed of the Monte Carlo streams (common random numbers across t)
    int ibd_max_coal;              /// keep states with at most this many within-pool coalescences (-1: all)
    int ibd_max_states;            /// automatic truncation if the full state space is larger
    bool ibd_no_symmetrize;        /// use the raw Monte Carlo generator (no reversible projection)
    int ibd_method;                /// propagation: -1 auto, 0 eigen, 1 uniformization
    bool ibd_full_posterior;       /// write the posterior of every colored IBD state
    string ibd_write_q;            /// write states, pi and Q to this file
    double ibd_cache_mb;           /// memory budget for precomputed emissions
    double ibd_t_tol;              /// golden-section tolerance on log(t)
    int ibd_precision;             /// digits in --ibd output files

    
    /// read relevant information
    void read_cmd_line ( int argc, char *argv[] );

};

#endif

