#pragma once

// Standard library includes
#include <vector>
#include <memory>
#include <random>

// Rcpp includes
#include <RcppArmadillo.h>
#include <progress.hpp>

// Local includes
#include "likelihoods.h"
#include "priors.h"
#include "sampler_types.h"


class VoronoiSampler {
  // Class members
  private:
    // Data
    arma::mat distance_matrix;
    size_t n_data;
    // Model definition
    std::shared_ptr<AbstractLikelihood> likelihood;
    std::shared_ptr<AbstractPrior> prior;
    // Algorithm Parameters
    AlgorithmParams algo_params;
    // Internal MCMC state
    TessellationState curr_state;
    // Random number generator
    std::mt19937 rng;

  // Public class methods
  public:
    // Constructor & destructor
    VoronoiSampler(const arma::mat & _distance_matrix, std::shared_ptr<AbstractLikelihood> _likelihood_ptr, std::shared_ptr<AbstractPrior> _prior_ptr, const AlgorithmParams & _algo_params);
    ~VoronoiSampler() = default;
    // Main MCMC loop
    MCMCOutput run();
    // Getters for results
    TessellationState get_current_state() const { return curr_state; }

  // Private class methods
  private:
    // Utilities (maybe compute_tessellation as an external function?)
    arma::uvec compute_tessellation(const std::vector<arma::uword> & centres) const;
    arma::vec apply_tempering(const arma::vec& log_probs, double tempering, const arma::uvec& valid_idx) const;
    // Proposal Generators
    arma::vec compute_birth_probs() const;
    arma::vec compute_death_probs() const;
    arma::vec compute_move_probs(int old_center_idx) const;
    // MCMC Steps
    void init();
    void birth_step();
    void death_step();
    void move_step();
    void step(size_t curr_iter); 
};