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
#include "multiview_py_sampler.h"
#include "rcpp_unpackers.h"
#include "voronoi_sampler.h"
#include "py_sampler.h"
#include "mcmc_utils.h"



//' Run Reversible Jump MCMC for Voronoi Tessellation
//' @title Run Reversible Jump MCMC for Voronoi Tessellation
//' @description This function runs a Reversible Jump Markov Chain Monte Carlo (RJMCMC) sampler
//' for a Voronoi tessellation-based clustering model.
//'
//' @param distance_matrix A square numeric matrix of pairwise distances between data points.
//' @param likelihood_params A list specifying the likelihood parameters. See \code{\link{likelihood_params}} for details.
//' @param prior_params A list specifying the prior parameters. See \code{\link{prior_params}} for details on the `"truncated_geometric"` prior.
//' @param algo_params A list of algorithm parameters. See \code{\link{algorithm_params}} for details on the tessellation model parameters.
//'
//' @return A list containing the MCMC output:
//'   \itemize{
//'     \item \code{cluster_allocs}: A matrix of cluster allocations for each retained sample.
//'     \item \code{centres}: A list of vectors, where each vector contains the indices of cluster centres for a retained sample.
//'     \item \code{n_clust}: A vector of the number of clusters for each retained sample.
//'     \item \code{lpdf}: A vector of the log-posterior density values for each retained sample.
//'     \item \code{iteration_time}: A vector of the iterations' execution time (in seconds) for each retained sample.
//'   }
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
        Rcpp::Named("lpdf") = out.lpdf,
        Rcpp::Named("iteration_time") = out.iteration_time
    );
};

//' Run Split-Merge MCMC for Pitman-Yor Process Mixture Model
//' @title Run Split-Merge MCMC for Pitman-Yor Process Mixture Model
//' @description This function runs a Split-Merge MCMC sampler for a Pitman-Yor process
//' mixture model.
//'
//' @param distance_matrix A square numeric matrix of pairwise distances between data points.
//' @param likelihood_params A list specifying the likelihood parameters. See \code{\link{likelihood_params}} for details.
//' @param prior_params A list specifying the Pitman-Yor process prior parameters. See \code{\link{prior_params}} for details on the `"PY-fixed"` and `"PY-hierarchical"` priors.
//' @param algo_params A list of algorithm parameters. See \code{\link{algorithm_params}} for details on the mixture model parameters.
//'
//' @return A list containing the MCMC output:
//'   \itemize{
//'     \item \code{cluster_allocs}: A matrix of cluster allocations for each retained sample.
//'     \item \code{n_clust}: A vector of the number of clusters for each retained sample.
//'     \item \code{discount}: A vector of the discount parameter values for each retained sample. For a fixed prior, this will be constant.
//'     \item \code{concentration}: A vector of the concentration parameter values for each retained sample. For a fixed prior, this will be constant.
//'     \item \code{lpdf}: A vector of the log-posterior density values for each retained sample.
//'     \item \code{iteration_time}: A vector of the iterations' execution time (in seconds) for each retained sample.
//'   }
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
    PYSampler sampler(distance_matrix, likelihood_ptr, prior_ptr, algo_cfg);
    MixtureMCMCOutput out = sampler.run();   
    // Return everything as a structured R List
    return Rcpp::List::create(
        Rcpp::Named("cluster_allocs") = out.cluster_allocs + 1,
        Rcpp::Named("n_clust") = out.n_clust,
        Rcpp::Named("discount") = out.discount,
        Rcpp::Named("concentration") = out.concentration,
        Rcpp::Named("lpdf") = out.lpdf,
        Rcpp::Named("iteration_time") = out.iteration_time
    );
};


//' Run Reversible Jump MCMC for Multi-View Voronoi Tessellation
//' @title Run Reversible Jump MCMC for Multi-View Voronoi Tessellation
//' @description This function runs an RJMCMC sampler for a multi-view Voronoi
//' tessellation-based clustering model, coupling multiple views through their clustering structures.
//'
//' @param distance_matrices A list of square numeric matrices, one for each view,
//'   representing pairwise distances.
//' @param likelihood_params A list containing parameters for the likelihood across all views. See \code{\link{multiview_params}} for details.
//' @param prior_params A list containing prior parameters for each view. See \code{\link{multiview_params}} for details.
//' @param algo_params A list of algorithm parameters. See \code{\link{algorithm_params}} for details on the tessellation model parameters.
//' @return A list containing the multi-view MCMC output:
//'   \itemize{
//'     \item \code{views}: A list of lists, where each inner list contains the MCMC output for a single view (see `mcmc_tessellation` return value).
//'     \item \code{joint_lpdf}: A vector of the joint log-posterior density values (including coupling) for each retained sample.
//'     \item \code{iteration_time}: A vector of the iterations' execution time (in seconds) for each retained sample.
//'   }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_tessellation_multiview(const Rcpp::List & distance_matrices, const Rcpp::List & likelihood_params, const Rcpp::List & prior_params, const Rcpp::List & algo_params) {
    // Export likelihood and prior parameters for each views
    Rcpp::List lik_views_params = likelihood_params["views_params"];
    Rcpp::List prior_views_params = prior_params["views_params"];
    // Check: size of likelihood_params and prior_params must be the same
    size_t n_views;
    if(lik_views_params.size() != prior_views_params.size()) {
        Rcpp::stop("Size of likelihood_params$views_params and prior_params$views_params must be the same");
    } else {
        n_views = lik_views_params.size();
    }
    // Safety check for distance matrices
    if (static_cast<size_t>(distance_matrices.size()) != n_views) {
        Rcpp::stop("The number of distance matrices must match the number of views");
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

//' Run Split-Merge MCMC for Multi-View Pitman-Yor Mixture Model
//' @title Run Split-Merge MCMC for Multi-View Pitman-Yor Mixture Model
//' @description This function runs a Split-Merge MCMC sampler for a multi-view Pitman-Yor
//' process mixture model, coupling multiple views.
//'
//' @param distance_matrices A list of square numeric matrices, one for each view.
//' @param likelihood_params A list containing parameters for the likelihood across all views. See \code{\link{multiview_params}} for details.
//' @param prior_params A list containing prior parameters for each view. See \code{\link{multiview_params}} for details.
//' @param algo_params A list of algorithm parameters. See \code{\link{algorithm_params}} for details on the mixture model parameters.
//' @return A list containing the multi-view MCMC output:
//'   \itemize{
//'     \item \code{views}: A list of lists, where each inner list contains the MCMC output for a single view (see `mcmc_PY` return value).
//'     \item \code{joint_lpdf}: A vector of the joint log-posterior density values (including coupling) for each retained sample.
//'     \item \code{iteration_time}: A vector of the iterations' execution time (in seconds) for each retained sample.
//'   }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_PY_multiview(const Rcpp::List & distance_matrices, const Rcpp::List & likelihood_params, const Rcpp::List & prior_params, const Rcpp::List & algo_params) {                     
    // Export likelihood and prior parameters for each view
    Rcpp::List lik_views_params = likelihood_params["views_params"];
    Rcpp::List prior_views_params = prior_params["views_params"];
    // Check: size of likelihood_params and prior_params must be the same
    size_t n_views;
    if(lik_views_params.size() != prior_views_params.size()) {
        Rcpp::stop("Size of likelihood_params$views_params and prior_params$views_params must be the same");
    } else {
        n_views = lik_views_params.size();
    }
    // Safety check for distance matrices
    if (static_cast<size_t>(distance_matrices.size()) != n_views) {
        Rcpp::stop("The number of distance matrices must match the number of views");
    }
    // Specify algorithm parameters (Using MixtureAlgorithmParams for PY)
    auto algo_cfg = Rcpp::as<MixtureAlgorithmParams>(algo_params);
    // Loop through the views and build model in each view
    std::vector<std::shared_ptr<PYSampler>> model_in_view;
    model_in_view.reserve(n_views);    
    for (size_t v = 0; v < n_views; ++v) {
        arma::mat curr_dist_matrix = Rcpp::as<arma::mat>(distance_matrices[v]);
        auto curr_likelihood = build_likelihood(lik_views_params[v]);
        auto curr_prior = build_mixture_prior(prior_views_params[v]);
        model_in_view.push_back(std::make_shared<PYSampler>(curr_dist_matrix, curr_likelihood, curr_prior, algo_cfg));
    }
    // Export coupling params from likelihood_params
    auto coupling_params = Rcpp::as<CouplingParams>(likelihood_params["coupling_params"]);
    // Construct MultiViewPYSampler object and run the MCMC
    MultiViewPYSampler sampler(model_in_view, coupling_params, algo_cfg);
    MultiViewMixtureMCMCOutput out = sampler.run();
    // Return everything as a structured R list
    return wrap_multiview_mixture_output(out);
};