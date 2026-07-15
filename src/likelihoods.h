#pragma once

// Rcpp includes
#include <RcppArmadillo.h>

// Abstract base class for likelihood objects
class AbstractLikelihood {
  public:
    // Virtual destructor
    virtual ~AbstractLikelihood() = default;
    // Evaluate the log-likelihood given the data and the cluster allocations
    virtual double eval_lpdf(const arma::mat& dist_matrix, const arma::uvec& cluster_allocs) const = 0;
};


// Derived concrete class: quadratic tessellation likelihood
// Struct to hold parameters for the quadratic tessellation likelihood
struct QuadraticLikelihoodParams {
    // Within-cluster parameters
    double shape_within, prior_shape_within, prior_rate_within;
    // Between-cluster parameters
    double shape_between, prior_shape_between, prior_rate_between;
    // Repulsion flag
    bool repulsion;
};

// Class implementation
class QuadraticTessellationLikelihood : public AbstractLikelihood {
  private:
    QuadraticLikelihoodParams params;
    bool debug = false; // Debug flag for internal logging
  public:
    // Constructor & Destructor
    QuadraticTessellationLikelihood(const QuadraticLikelihoodParams & _params) : params(_params) {};
    ~QuadraticTessellationLikelihood() = default;
    // Evaluation function (override)
    double eval_lpdf(const arma::mat & dist_matrix, const arma::uvec & cluster_allocs) const override;
};