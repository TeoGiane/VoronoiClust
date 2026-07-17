// Standard library includes
#include <memory>

// Rcpp includes and directives
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
// [[Rcpp::plugins(openmp)]]

// Local includes
#include "factory.h"
#include "rcpp_unpackers.h"
#include "voronoi_sampler.h"
#include "py_sampler.h"


//' Run Reversible Jump MCMC for Spatial Voronoi Tessellation
//'
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_tessellation(const arma::mat& distance_matrix, Rcpp::List likelihood_params, Rcpp::List prior_params, Rcpp::List algo_params) {
    // Instantiate the likelihood and prior objects using the factory functions
    auto likelihood_ptr = build_likelihood(likelihood_params);
    auto prior_ptr = build_prior(prior_params);
    // Specify algorithm parameters
    auto algo_cfg = Rcpp::as<AlgorithmParams>(algo_params);
    // Construct VoronoiSampler object and run the MCMC
    VoronoiSampler sampler(distance_matrix, likelihood_ptr, prior_ptr, algo_cfg);
    MCMCOutput out = sampler.run();   
    // Convert the vector of vectors into an R List of numeric vectors
    Rcpp::List r_centres(out.centres.size());
    for(size_t i = 0; i < out.centres.size(); ++i) {
        // Convert to arma::uvec so we can perform vectorized addition
        arma::uvec r_idx = arma::conv_to<arma::uvec>::from(out.centres[i]) + 1;
        r_centres[i] = r_idx;
    }
    // Return everything as a structured R List
    return Rcpp::List::create(
        Rcpp::Named("cluster_allocs") = out.cluster_allocs + 1,
        Rcpp::Named("centres") = r_centres,
        Rcpp::Named("n_clust") = out.n_clust,
        Rcpp::Named("lpdf") = out.lpdf
    );
};

//' Run Split-Merge MCMC for Pitman-Yor Process Mixture Model
//'
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_PY(const arma::mat& distance_matrix, Rcpp::List likelihood_params, Rcpp::List prior_params, Rcpp::List algo_params) {                  
    // Instantiate the likelihood and prior objects using the factory functions
    auto likelihood_ptr = build_likelihood(likelihood_params);
    auto prior_ptr = build_mixture_prior(prior_params);
    // Specify algorithm parameters
    auto algo_cfg = Rcpp::as<MixtureAlgorithmParams>(algo_params);
    // Construct PYSplitMergeSampler object and run the MCMC
    PYSplitMergeSampler sampler(distance_matrix, likelihood_ptr, prior_ptr, algo_cfg);
    MixtureMCMCOutput out = sampler.run();   
    // Return everything as a structured R List
    return Rcpp::List::create(
        Rcpp::Named("cluster_allocs") = out.cluster_allocs + 1,
        Rcpp::Named("n_clust") = out.n_clust,
        Rcpp::Named("discount") = out.discount,
        Rcpp::Named("concentration") = out.concentration,
        Rcpp::Named("lpdf") = out.lpdf
    );
};