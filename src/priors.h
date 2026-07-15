#pragma once

// Standard library includes
#include <cmath>

// Rcpp includes
#include <RcppArmadillo.h>


// Abstract base class for priors
class AbstractPrior {
  public:
    // Virtual destructor
    virtual ~AbstractPrior() = default;
    // Evaluate the prior given the number of clusters
    virtual double eval_lpdf(unsigned int k) const = 0;
};


// Derived concrete class: truncated geometric prior
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
    TruncatedGeometricPrior(const TruncatedGeometricParams & _params) : params(_params) {}
    ~TruncatedGeometricPrior() override = default;
    // Evaluation function (override)
    double eval_lpdf(unsigned int k) const override;
};