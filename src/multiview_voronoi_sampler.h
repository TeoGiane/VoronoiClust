#pragma once

// Standard library includes
#include <vector>
#include <memory>
#include <random>
#include <cmath>

// Rcpp includes
#include <RcppArmadillo.h>
#include <progress.hpp>
#include <gsl/gsl_sf_hyperg.h>

// Local includes
#include "mcmc_utils.h"
#include "sampler_types.h"
#include "voronoi_sampler.h"


// Struct to hold Multi View Likelihood parameters
// template <typename T>
// struct MultiViewLikelihoodParams {
//     std::vector<T> view_likelihood_params;
//     CouplingParams coupling_params;
// };

// Struct to hold Multi View Prior parameters
// template <typename T>
// struct MultiViewPriorParams {
//     std::vector<T> view_prior_params;
// };


class MultiViewVoronoiSampler {
  // Class members
  private:
    // Data
    int n_views;
    // Models definition
    std::vector<std::shared_ptr<VoronoiSampler>> model_in_view;
    CouplingParams coupling_params;
    // Storage for couplng terms
    double coupling_log_const;
    arma::mat pairwise_couplings;
    double total_coupling;
    // Algorithm parameters
    AlgorithmParams algo_params;
    // Internal MCMC state
    // MultiViewTessellationState curr_state;
    // Random number generator
    std::mt19937 rng;
  
  // Public methods
  public:
    // Constructor
    MultiViewVoronoiSampler(const std::vector<std::shared_ptr<VoronoiSampler>>& _model_in_view,
                            const CouplingParams& _coupling_params,
                            const AlgorithmParams & _algo_params);
    // Main execution loop
    MultiViewMCMCOutput run();
  
  // Private methods
  private:
    // Multi-view helper methods
    ConditionalCouplingResult compute_conditional_coupling(int curr_view, const arma::uvec& target_z) const;
    // MCMC steps
    void init();
    void step(size_t curr_iter);
};