#include "priors.h"

/* Truncated Geometric class */
TruncatedGeometricPrior::TruncatedGeometricPrior(const TruncatedGeometricParams & _params) : params(_params) {};

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
};

/* Pitman Yor fixed prior */
PYFixedPrior::PYFixedPrior(const PYFixedParams & params) : discount(params.discount), concentration(params.concentration) {
    if(discount < 0.0 || discount >= 1.0) {
        throw std::invalid_argument("Pitman-Yor discount parameter must be in [0, 1).");
    }
    if(concentration <= -discount) {
        throw std::invalid_argument("Pitman-Yor concentration parameter must be strictly greater than -discount.");
    }
};

double PYFixedPrior::eval_lpdf(const arma::uvec & clust_allocs) const {
    size_t n_data = clust_allocs.n_elem;
    if (n_data == 0) return 0.0;
    // Get cluster sizes
    int max_id = clust_allocs.max();
    arma::uvec sizes = arma::zeros<arma::uvec>(max_id + 1);
    int n_clust = 0;    
    for (size_t i = 0; i < n_data; ++i) {
        if (sizes(clust_allocs(i))++ == 0) {
            n_clust++;
        }
    }
    // Compute eppf
    double log_prob = 0.0;
    for (int i = 1; i < n_clust; ++i) {
        log_prob += std::log(concentration + i * discount);
    }
    log_prob += std::lgamma(concentration + 1.0) - std::lgamma(concentration + n_data);
    double log_gamma_1_minus_d = std::lgamma(1.0 - discount);      
    
    for (arma::uword k = 0; k < sizes.n_elem; ++k) {
        int n_c = sizes(k);
        if (n_c > 0) {
            log_prob += std::lgamma(n_c - discount) - log_gamma_1_minus_d;
        }
    }
    return log_prob;
};

double PYFixedPrior::eval_pred_lpdf_existing(int n_k_minus_i) const {
    return std::log(n_k_minus_i - discount);
};

double PYFixedPrior::eval_pred_lpdf_new(int K_minus_i) const {
    return std::log(concentration + K_minus_i * discount);
};

/* Pitman Yor hierarchical prior */
PYHierarchicalPrior::PYHierarchicalPrior(const PYHierarchicalParams & params) : PYFixedPrior(PYFixedParams{params.discount_alpha / (params.discount_alpha + params.discount_beta), params.concentration_shape / params.concentration_rate}), hyper_params(params) {
    // Compatibility checks
    if (hyper_params.discount_alpha <= 0.0 || hyper_params.discount_beta <= 0.0 || hyper_params.concentration_shape <= 0.0 || hyper_params.concentration_rate <= 0.0) {
        throw std::invalid_argument("Hyper-parameters in the PY Hierarchical Prior must be strictly positive.");
    }
};
