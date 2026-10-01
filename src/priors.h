#pragma once

// Standard library includes
#include <cmath>

// Rcpp includes
#include <RcppArmadillo.h>


/* Abstract base class for tessellation priors */
class AbstractPrior {
  public:
    // Virtual destructor
    virtual ~AbstractPrior() = default;
    // Evaluate the prior given the number of clusters
    virtual double eval_lpdf(unsigned int k) const = 0;
};


/* Truncated Geometric prior */
// Struct to hold parameters for the truncated geometric prior
struct TruncatedGeometricParams {
    unsigned int min;
    unsigned int max;
    double prob;
};

// Class implementation
class TruncatedGeometricPrior : public AbstractPrior {
  private:
    TruncatedGeometricParams params;
    bool debug = false; // Debug flag for internal logging
  public:
    // Constructor & Destructor
    TruncatedGeometricPrior(const TruncatedGeometricParams & _params);
    ~TruncatedGeometricPrior() override = default;
    // Evaluation function
    double eval_lpdf(unsigned int k) const override;
};


/* Abstract base class for mixture priors */
class AbstractMixturePrior {
  public:
    // Virtual destructor
    virtual ~AbstractMixturePrior() = default;
    // Evaluate the prior given the number of clusters
    virtual double eval_lpdf(const arma::uvec & clust_allocs) const = 0;
};


/* Pitman-Yor fixed prior */
// Struct to hold parameters
struct PYFixedParams {
	double discount, concentration;
};

// Class implementation
class PYFixedPrior : public AbstractMixturePrior {
  protected: 
    // State variables (separate due to inheritance)
    double discount;
    double concentration;
    bool debug = false;
  public:
    // Constructor & Destructor
    PYFixedPrior(const PYFixedParams & params);
    ~PYFixedPrior() override = default;
    // Evaluation function (override)
    double eval_lpdf(const arma::uvec & clust_allocs) const override;
    // Auxiliary functions for Gibbs steps
    double eval_pred_lpdf_existing(int n_k_minus_i) const;
    double eval_pred_lpdf_new(int K_minus_i) const;
    // Getters and setters
    void set_discount(double _discount) { discount = _discount; };
    void set_concentration(double _concentration) { concentration = _concentration; };
    double get_discount() const { return discount; };
    double get_concentration() const { return concentration; };
};

/* Pitman-Yor Hierarchical prior */
// Struct to hold parameters
struct PYHierarchicalParams {
	// Beta prior for discount
	double discount_alpha, discount_beta;
	// Gamma prior for concentration
    double concentration_shape, concentration_rate;
};

// Class implementation
class PYHierarchicalPrior : public PYFixedPrior {
  private:
    PYHierarchicalParams hyper_params;
  public:
    // Constructor & Destructor
    PYHierarchicalPrior(const PYHierarchicalParams & params);
    ~PYHierarchicalPrior() override = default;
    // Getter for the hyper-parameters
    PYHierarchicalParams get_hyper_params() const { return hyper_params; };
};