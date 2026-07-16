#pragma once

// Rcpp includes
#include <RcppArmadillo.h>


// Struct that specify the algorithm parameters
struct AlgorithmParams {
    int iterations;
    int burnin;
    int thinning;
    double tempering = 0;
    int init_n_clust = 5;
    int random_seed = 20260714;
    bool debug = false;
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


/* ALGORITHM PARAMETERS */

/* END */