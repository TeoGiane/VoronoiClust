#pragma once

// Standard library includes
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <chrono>
#include <sstream>
#include <stdexcept>

// Rcpp includes
#include <RcppArmadillo.h>
#include <progress.hpp>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_errno.h>

// Local includes
#include "contingency_tracker.h"
#include "sampler_types.h"
#include "voronoi_sampler.h"

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
    double total_coupling;
    ContingencyTracker tracker;
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
                            const AlgorithmParams& _algo_params);
    ~MultiViewVoronoiSampler() = default;
    // Main execution loop
    MultiViewMCMCOutput run();

  // Private methods
  private:
    void init();
    double pair_coupling_energy(double rand_index) const;
    double conditional_coupling_from_tracker(size_t view) const;
    void compute_total_coupling();
    double score_proposal_coupling(size_t view, const arma::uvec& proposed_allocs);
    void apply_allocation_changes(size_t view, const arma::uvec& target_allocs);
    void step(size_t curr_iter);
};
