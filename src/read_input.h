#ifndef __READ_INPUT_H
#define __READ_INPUT_H

#include "ibd_utils.h"
#include <fstream>

void read_file ( cmd_line &options, vector<markov_chain> &markov_chain_information, map<int,vector<vector<int> > > &state_list, vector<int> &position, vector<double> &recombination_rate, vector<string> &chromosomes ) {
    
    /// vector to hold index of inbred path if we have variable ploidy
    vector<int> path_index( markov_chain_information.size(), 0 ) ;
    
    /// stream in file
    ifstream in ( options.input_file.c_str() ) ;
    
    //// since the first site transition matrix does not matter, we can print anything
    double extra_recombination = 1 ;
    string last_chrom = "" ;
    
    while( !in.eof() ) {
        
        input_line new_line ;
        in >> new_line.chrom >> new_line.pos ;
        
        /// if two adjacent sites have the same positions, skip second
        if ( ( position.size() > 0 && new_line.pos == position.back() ) ) {
            getline( in, new_line.chrom ) ;
            continue ;
        }
                        
        // read reference panel genotype counts
        new_line.reference_counts.resize( options.ancestry_proportion.size() ) ;
        int count = 0 ;
        for ( int p = 0 ; p < options.ancestry_proportion.size() ; p ++ ) {
            double count1, count2 ;
            in >> count1 >> count2 ;
            new_line.reference_counts[p].push_back(count1) ;
            new_line.reference_counts[p].push_back(count2) ;
            new_line.reference_counts[p].push_back(count1+count2) ;

	/// note that this subsampling approach assumes this is only necessary for human data. which for now is the only plausible dataset with reference population sizes thi slarge. 
            if ( new_line.reference_counts[p][2] > 998 ) { 
                subsample_reads( new_line.reference_counts[p][0], new_line.reference_counts[p][1], 998 ) ;
		new_line.reference_counts[p][2] = new_line.reference_counts[p][0] + new_line.reference_counts[p][1] ; 
            }
        }
        
        /// read recombination rate
        in >> new_line.recombination_rate ;
        
        /// if line specific error rates are provided
        new_line.error_1 = options.error_rate ;
        new_line.error_2 = options.error_rate ;
        if ( options.error_rates == true ) {
            in >> new_line.error_1 >> new_line.error_2 ;
        }
        
        // read sample panel read counts
        new_line.sample_counts.resize( markov_chain_information.size() ) ;
        for ( int m = 0 ; m < markov_chain_information.size() ; m ++ ) {
            double count1, count2 ;
            in >> count1 >> count2 ;
            
            /// subsample reads to a maximum depth so we can compute multinomial probs without overflow errors
            if ( count1 + count2 > 170 ) {
                subsample_reads( count1, count2, 170 ) ;
            }
            
            /// now store counts and total for sample
            new_line.sample_counts[m].push_back(count1) ;
            new_line.sample_counts[m].push_back(count2) ;
            new_line.sample_counts[m].push_back(count1+count2) ;
        }
        
        if ( new_line.chrom != last_chrom ) {
            recombination_rate.push_back( 0.5 ) ;
            last_chrom = new_line.chrom ;
            extra_recombination = 0 ; 
        }

        /// ignore lines where recombination may not be suffiicent to make sites independent
        /// this might be useful in place of LD pruning
        else {

            extra_recombination += new_line.recombination_rate ;
            if ( extra_recombination < options.minimum_distance ) {
                continue ;
            }
            new_line.recombination_rate = extra_recombination ;
            extra_recombination = 0 ;
        
            recombination_rate.push_back( new_line.recombination_rate/ ( new_line.pos - position.back() ) ) ;
        }

        /// record position
        position.push_back( new_line.pos ) ;
        chromosomes.push_back( new_line.chrom ) ;

        /// check all path indexes and update as needed
        for ( int m = 0 ; m < markov_chain_information.size() ; m ++ ) {
            
            if ( markov_chain_information[m].path_file != "null" ) {
                
                /// record previous ploidy
                int previous_ploidy = markov_chain_information[m].sample_ploidy_path[path_index[m]].ploidy ;
                
                /// check to make sure we're on the right ploidy tract
                while ( new_line.chrom != markov_chain_information[m].sample_ploidy_path[path_index[m]].chrom ) {
                    path_index[m] ++ ;
                }
                while ( new_line.pos > markov_chain_information[m].sample_ploidy_path[path_index[m]].stop ) {
                    path_index[m] ++ ;
                }
                
                /// record switches
                if ( previous_ploidy != markov_chain_information[m].sample_ploidy_path[path_index[m]].ploidy ) {
                    
                    markov_chain_information[m].ploidy_switch_position.push_back( position.size() - 1 ) ;
                    markov_chain_information[m].ploidy_switch.push_back( markov_chain_information[m].sample_ploidy_path[path_index[m]].ploidy ) ;
                }
            }
        }
    
        /// 
        if ( options.genotype == false ) {
            for ( int m = 0 ; m < markov_chain_information.size() ; m ++ ) {
                vec emissions ;
                create_emissions_matrix( markov_chain_information[m].sample_ploidy_path[path_index[m]].ploidy, new_line, options.ancestral_fixed, state_list[markov_chain_information.at(m).sample_ploidy_path[path_index[m]].ploidy], m, options.ancestry_pulses, emissions ) ;
                markov_chain_information[m].emission_probabilities.push_back( emissions ) ;
            }
        }
        
        /// create emissions matrix with genotypes
        else {
            for ( int m = 0 ; m < markov_chain_information.size() ; m ++ ) {
                vec emissions ;
                create_emissions_matrix_genotype( markov_chain_information[m].sample_ploidy_path[path_index[m]].ploidy, new_line, options.ancestral_fixed, state_list[markov_chain_information.at(m).sample_ploidy_path[path_index[m]].ploidy], m, options.ancestry_pulses, emissions ) ;
                markov_chain_information[m].emission_probabilities.push_back( emissions ) ;
            }
        }
    }
    
    /// to avoid lookahead errors
    for ( int m = 0 ; m < markov_chain_information.size() ; m ++ ) {
        markov_chain_information[m].ploidy_switch_position.push_back( position.size() ) ;
    }
}


namespace ibd {

class ibd_data {
public:
  std::vector<std::string> chrom;
  std::vector<int> pos;
  std::vector<double> dist;               // Morgans from the previous retained site of the same chromosome
  std::vector<size_t> block_start;        // first site of each chromosome, plus a final sentinel = n_sites
  std::vector<double> ref;                // 4 per site for 2 ancestry populations: C0A, C0a, C1A, C1a
  std::vector<double> err1;
  std::vector<double> err2;
  std::vector<std::vector<double> > counts;   // counts[sample][2*site + {0,1}] = (A, a)

  size_t n_sites() const { return pos.size(); }
  size_t n_blocks() const { return block_start.empty() ? 0 : block_start.size() - 1; }
};

inline void read_ibd_input(
  const std::string &file,
  int n_ancestry,
  int n_samples,
  double minimum_distance,
  bool site_errors,
  double error_rate,
  ibd_data &data
) {
  if (n_ancestry != 2) {
    error("IBD mode currently only supports 2 ancestry populations");
  }

  std::ifstream in(file.c_str());
  if (!in) error("cannot open input file " + file);

  data.counts.assign(n_samples, std::vector<double>());
  
  std::string line;
  std::string last_chrom;
  double extra = 0.0;
  size_t line_no = 0;

  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty()) continue;

    std::vector<std::string> fields = split(line, '\t');
  
    size_t i = 0;
    std::string ch = fields[i++];
    int p = std::stoi(fields[i++]);
  
    std::vector<double> ref_counts(2 * n_ancestry);
    for (double &ref_count : ref_counts) {
      ref_count = std::stod(fields[i++]);
    }
  
    double morgan = std::stod(fields[i++]);
    double e1 = error_rate;
    double e2 = error_rate;
    if (site_errors) {
      e1 = std::stod(fields[i++]);
      e2 = std::stod(fields[i++]);
    }
  
    std::vector<double> sample_counts(2 * n_samples);
    for (double &sample_count : sample_counts) {
      sample_count = std::stod(fields[i++]);
    }
  
    bool new_chrom = ch != last_chrom;

    // Skip duplicate positions.
    if (!new_chrom && !data.pos.empty() && p == data.pos.back()) {
      std::cerr << "Warning: duplicate position " << p << " on chromosome " << ch << " at line " << line_no << " ... First occurrence is kept." << std::endl;
      continue;
    }

    double distance = 0.0;
    if (new_chrom) {
      last_chrom = ch;
      extra = 0.0;
      data.block_start.push_back(data.pos.size());
    } else {
      extra += morgan;
      if (extra < minimum_distance) continue; // Skip sites closer than minimum_distance to the previous retained site.

      distance = extra;
      extra = 0.0;
    }

    // Add data to the data object.
    data.chrom.push_back(ch);
    data.pos.push_back(p);
    data.dist.push_back(distance);
    data.ref.insert(data.ref.end(), ref_counts.begin(), ref_counts.end());
    data.err1.push_back(e1);
    data.err2.push_back(e2);
    for (int s = 0; s < n_samples; s ++) {
      data.counts[s].push_back(sample_counts[2 * s]);
      data.counts[s].push_back(sample_counts[2 * s+1]);
    }
  }
  data.block_start.push_back(data.pos.size());
  if (data.pos.empty()) {
    error("no sites read from " + file);
  }
}

}

#endif

