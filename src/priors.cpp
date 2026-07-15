#include "priors.h"

double TruncatedGeometricPrior::eval_lpdf(unsigned int k) const {
    // Enforce the truncated support [min_k, max_k] strictly!
    if (k < params.min || k > params.max) {
        return -arma::datum::inf; // Impossible configuration -> 0 probability
    }
    
    // Geometric Prior on K
    double log_geom = std::log(params.prob) + k * std::log(1.0 - params.prob);
    
    // Combinatorial penalty (n choose k) for truncation
    double n = static_cast<double>(params.max);
    double log_binom = std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
                        
    return log_geom - log_binom;
}