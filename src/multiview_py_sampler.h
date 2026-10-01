#pragma once

// Standard library includes
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <functional>
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
#include "py_sampler.h"


class MultiViewPYSampler {
  // Class members
  private:
    // Data
    size_t n_views;
    // Models definition
    std::vector<std::shared_ptr<PYSampler>> model_in_view;
    CouplingParams coupling_params;
    // Storage for coupling terms
    double coupling_log_const;
    double total_coupling;
    ContingencyTracker tracker;
    // Algorithm parameters
    MixtureAlgorithmParams algo_params;
    // Random number generator
    std::mt19937 rng;
  
  // Class methods
  public:
    MultiViewPYSampler(const std::vector<std::shared_ptr<PYSampler>> & _model_in_view,
                       const CouplingParams & _coupling_params,
                       const MixtureAlgorithmParams & _algo_params);
    ~MultiViewPYSampler() = default;
    MultiViewMixtureMCMCOutput run();

  // Private class methods
  private:
    // Sampler initialization 
    void init();
    // Maps a Rand index to the coupling ENERGY contributed by one pair of
    // views. Centralises the degenerate-argument clamping, the GSL call and
    // its error handling, and -- critically -- the SIGN convention: PYSampler
    // SUBTRACTS the coupling everywhere, so what it receives must be an
    // energy (low = views agree), not the log-density that U() gives.
    double pair_coupling_energy(double rand_index) const;
    // Calculates the conditional coupling using the raw naive math (used for Split/Merge block proposals)
    double compute_conditional_coupling(size_t curr_view, const arma::uvec& target_z) const;
    // Compute current coupling using the tracker variables
    double compute_current_coupling(size_t v) const;
    // Compute total coupling from scratch using the tracker variables
    void compute_total_coupling();
    // Single MCMC iteration
    void step(size_t curr_iter);
};