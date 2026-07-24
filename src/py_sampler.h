#pragma once

// Standard library includes
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <functional>

// Rcpp includes
#include <RcppArmadillo.h>
#include <progress.hpp>

// Local includes
#include "likelihoods.h"
#include "priors.h"
#include "sampler_types.h"
#include "mcmc_utils.h"


// Callback types defintions (for multi-view extension)
// Full state evaluation (used for Split/Merge)
using FullCouplingCallback = std::function<double(const arma::uvec &)>;
// O(1) Gibbs evaluation: inputs are (obs_i, old_clust, new_clust)
using GibbsCouplingCallback = std::function<double(int, int, int)>;
// O(1) Gibbs state update: inputs are (obs_i, old_clust, new_clust)
using GibbsUpdateCallback = std::function<void(int, int, int)>;



class PYSampler {
  // Class members
  private:
    // Data
    arma::mat distance_matrix;
    size_t n_data;
    // Model definition
    std::shared_ptr<AbstractLikelihood> likelihood;
    std::shared_ptr<AbstractMixturePrior> prior;
    // Algorithm Parameters
    MixtureAlgorithmParams algo_params;
    // Internal MCMC state
    MixtureState curr_state;
	// Initial adaptive RWMH sds
	double discount_sd = 0.1;
	double concentration_sd = 0.1;
    // Random number generator
    std::mt19937 rng;

  // Public class methods
  public:
    // Constructor & destructor
    PYSampler(const arma::mat & _distance_matrix,
						std::shared_ptr<AbstractLikelihood> _likelihood_ptr,
						std::shared_ptr<AbstractMixturePrior> _prior_ptr,
						const MixtureAlgorithmParams & _algo_params);
    ~PYSampler() = default;
    // Proper initialization method
    void init();
    // Step method (exposed for milti-view extension)
    void step(size_t curr_iter, FullCouplingCallback full_coupling_cb = nullptr, GibbsCouplingCallback gibbs_coupling_cb = nullptr,
              GibbsUpdateCallback gibbs_update_cb = nullptr); 
    // Main MCMC loop
    MixtureMCMCOutput run();
    // Getters for results
    MixtureState get_current_state() const { return curr_state; }

  // Private class methods
  private:
    // MCMC Steps
    void split_step(int obs_i, int obs_j, FullCouplingCallback full_coupling_cb = nullptr);
    void merge_step(int obs_i, int obs_j, FullCouplingCallback full_coupling_cb = nullptr);
	void gibbs_step(GibbsCouplingCallback gibbs_coupling_cb = nullptr, GibbsUpdateCallback gibbs_update_cb = nullptr);
	void sample_discount(size_t curr_iter);
	void sample_concentration(size_t curr_iter);
    // Utilities
    arma::uvec standardize_allocs(const arma::uvec & allocs) const;
	double restricted_gibbs_sweep(arma::uvec & allocs, const arma::uvec & members,
								  unsigned int obs_i, unsigned int obs_j,
								  unsigned int clust_i, unsigned int clust_j,
								  bool sample, const arma::uvec & target_allocs);

};