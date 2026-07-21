// Standard library includes
#include <memory>

// Rcpp includes and directives
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]
// [[Rcpp::depends(RcppProgress)]]
// [[Rcpp::plugins(openmp)]]

// Local includes
#include "factory.h"
#include "multiview_voronoi_sampler.h"
#include "rcpp_unpackers.h"
#include "voronoi_sampler.h"
#include "py_sampler.h"
#include "mcmc_utils.h"



//' Run Reversible Jump MCMC for Voronoi Tessellation
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


//' Run Reversible Jump MCMC for Multi-View Voronoi Tessellation
//'
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_multiview_tessellation(const Rcpp::List & distance_matrices, const Rcpp::List & likelihood_params, const Rcpp::List & prior_params, const Rcpp::List & algo_params) {
    // Export likelihood and prior parameters for each views
    Rcpp::List lik_views_params = likelihood_params["views_params"];
    Rcpp::List prior_views_params = prior_params["views_params"];
    // Check: size of likelihood_params and prior_params must be the same
    size_t n_views;
    if(lik_views_params.size() != prior_views_params.size()) {
        throw std::invalid_argument("Size of likelihood_params$views_params and prior_params$views_params must be the same");
    } else {
        n_views = lik_views_params.size();
    }
    // Specify algorithm parameters
    auto algo_cfg = Rcpp::as<AlgorithmParams>(algo_params);
    // Loop through the views and build model in each view
    std::vector<std::shared_ptr<VoronoiSampler>> model_in_view;
    model_in_view.reserve(n_views);
    for (size_t v = 0; v < n_views; ++v) {
        arma::mat curr_dist_matrix = Rcpp::as<arma::mat>(distance_matrices[v]);
        auto curr_likelihood = build_likelihood(lik_views_params[v]);
        auto curr_prior = build_prior(prior_views_params[v]);
        model_in_view.push_back(std::make_shared<VoronoiSampler>(curr_dist_matrix, curr_likelihood, curr_prior, algo_cfg));
    }
    // Export coupling params from likelhood_params
    auto coupling_params = Rcpp::as<CouplingParams>(likelihood_params["coupling_params"]);
    // Construct MultiViewVoronoiSampler object and run the MCMC
    MultiViewVoronoiSampler sampler(model_in_view, coupling_params, algo_cfg);
    MultiViewMCMCOutput out = sampler.run();
    // Return everything as a structured R list
    return wrap_multiview_output(out);
};