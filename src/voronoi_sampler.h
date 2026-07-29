#pragma once

// Standard library includes
#include <vector>
#include <memory>
#include <random>
#include <chrono>

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
    VoronoiSampler(const arma::mat & _distance_matrix,
                   std::shared_ptr<AbstractLikelihood> _likelihood_ptr,
                   std::shared_ptr<AbstractPrior> _prior_ptr,
                   const AlgorithmParams & _algo_params);
    ~VoronoiSampler() = default;
    // Proper initialization method
    void init();
    // Main MCMC loop
    MCMCOutput run();
    // Public proposal generator (useful for MultiView version)
    TessellationProposal generate_proposal(size_t curr_iter);
    // Force apply a state update (useful for MultiView version)
    void apply_accepted_proposal(const TessellationProposal& prop);
    // Getters for results
    TessellationState get_current_state() const { return curr_state; };
    std::shared_ptr<AbstractPrior> get_prior() const { return prior; };

  // Private class methods
  private:
    // Utilities (maybe compute_tessellation as an external function?)
    arma::uvec compute_tessellation(const std::vector<arma::uword> & centres) const;
    arma::vec apply_tempering(const arma::vec& log_probs, double tempering, const arma::uvec& valid_idx) const;
    // Probability Generators
    arma::vec compute_birth_probs() const;
    arma::vec compute_death_probs() const;
    arma::vec compute_move_probs(int old_center_idx) const;
    // Proposal generators
    TessellationProposal generate_birth_proposal();
    TessellationProposal generate_death_proposal();
    TessellationProposal generate_move_proposal();
    // Proposal tester
    void test_proposal(const TessellationProposal & prop_state);
    // MCMC Step
    void step(size_t curr_iter); 
};