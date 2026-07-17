#pragma once

// Rcpp includes
#include <RcppArmadillo.h>

/* Sampler types for Tessellation samplers */
// Struct that specify the algorithm parameters
struct AlgorithmParams {
    size_t iterations;
    size_t burnin;
    size_t thinning;
    int init_n_clust = 5;
    double tempering = 0;
    int random_seed = 20260714;
    bool debug = false;
};

// Struct representing the proposal of a tessellation MCMC sampler
struct TessellationProposal {
    int prop_n_clust;
    arma::uvec prop_cluster_allocs;
    std::vector<arma::uword> prop_centres;
    double prop_lpdf;
    double prob_new_old;
    double prob_old_new;
    int centre_to_add = -1;
    int centre_to_remove = -1;
};

// Struct representing the MCMC state at each iteration
struct TessellationState {
    int n_clust;
    arma::uvec cluster_allocs;
    arma::uvec is_centre;
    std::vector<arma::uword> cluster_centres;
    double lpdf;
    // Debug function for print current state of the chain
    void print() const {
        Rcpp::Rcout << "n_clust: " << n_clust << std::endl;
        Rcpp::Rcout << "cluster_allocs: " << cluster_allocs.t() << std::endl;
        Rcpp::Rcout << "is_centre: " << is_centre.t() << std::endl;
        Rcpp::Rcout << "cluster_centres: ";
        for (const auto& c : cluster_centres) { Rcpp::Rcout << c << " "; }
        Rcpp::Rcout << std::endl;
        Rcpp::Rcout << "lpdf: " << lpdf << std::endl;
    }
};

// Struct that holds the actual MCMC output
struct MCMCOutput {
    arma::umat cluster_allocs;                      // S rows x N columns
    std::vector<std::vector<arma::uword>> centres;  // Length S
    arma::uvec n_clust;                             // Length S
    arma::vec lpdf;                                 // Length S
};


/* Sampler types for traditional mixtures samplers */
// Struct that specify the algorithm parameters
struct MixtureAlgorithmParams {
    size_t iterations;
    size_t burnin;
    size_t thinning;
    int init_n_clust = 5;
	// Jain & Neal sweeps
	int n_sweeps = 1;
	// Adaptive MH
    double target_acc_rate = 0.44;
    double adapt_decay = 0.6;
	// reproducibility & debug
	int random_seed = 20260714;
	bool debug = false;
};

// Struct representing the MCMC state at each iteration
struct MixtureState {
    int n_clust;
    arma::uvec cluster_allocs;
	double discount;
	double concentration;
    double lpdf;
    // Debug function for print current state of the chain
    void print() const {
        Rcpp::Rcout << "n_clust: " << n_clust << std::endl;
        Rcpp::Rcout << "cluster_allocs: " << cluster_allocs.t() << std::endl;
		Rcpp::Rcout << "discount: " << discount << std::endl;
		Rcpp::Rcout << "concentration: " << concentration << std::endl;
        Rcpp::Rcout << "lpdf: " << lpdf << std::endl;
    };
};

// Struct that holds the actual MCMC output
struct MixtureMCMCOutput {
    arma::umat cluster_allocs;                      // S rows x N columns
    arma::uvec n_clust;                             // Length S
	arma::vec discount;                             // Length S
	arma::vec concentration;                        // Length S
    arma::vec lpdf;                                 // Length S
};