#pragma once

#include <optional>
// Rcpp includes
#include <RcppArmadillo.h>

// Abstract base class for likelihood objects
class AbstractLikelihood {
  public:
    // Virtual destructor
    virtual ~AbstractLikelihood() = default;
    // Evaluate the log-likelihood given the data and the cluster allocations
    virtual double eval_lpdf(const arma::mat& dist_matrix, const arma::uvec& cluster_allocs, const std::optional<const std::vector<arma::uword>>& centers = std::nullopt) const = 0;
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
    double eval_lpdf(const arma::mat & dist_matrix, const arma::uvec & cluster_allocs, const std::optional<const std::vector<arma::uword>>& centers = std::nullopt) const override;
    // Getter (needed by the incremental sufficient-statistics cache)
    const QuadraticLikelihoodParams & get_params() const { return params; };
};

// Derived concrete class: linear tessellation likelihood
// Struct to hold parameters for the linear tessellation likelihood
struct LinearLikelihoodParams {
    // Within-cluster parameters (Gamma-Gamma conjugate)
    double shape_within, prior_shape_within, prior_rate_within;
    // Between-cluster parameters (Gamma likelihood)
    double shape_between, rate_between;
    // Repulsion flag
    bool repulsion;
};

// Class implementation
class LinearTessellationLikelihood : public AbstractLikelihood {
  private:
    LinearLikelihoodParams params;
  public:
    LinearTessellationLikelihood(const LinearLikelihoodParams & _params) : params(_params) {};
    ~LinearTessellationLikelihood() = default;
    double eval_lpdf(const arma::mat & dist_matrix, const arma::uvec & cluster_allocs, const std::optional<const std::vector<arma::uword>>& centers = std::nullopt) const override;
    // Getter (needed by the incremental sufficient-statistics cache)
    const LinearLikelihoodParams & get_params() const { return params; };
};

/* Extra likelihood parameters for multi-view models */
// Struct to hold multi-view coupling parameters
struct CouplingParams {
    double strength_alpha;
    double strength_beta;
};

// Helper struct to manage computation of conditional couplings between views
struct ConditionalCouplingResult {
    double total_conditional_sum;
    std::vector<double> pairwise_terms;
};
