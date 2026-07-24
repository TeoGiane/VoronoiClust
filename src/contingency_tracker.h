#pragma once

// Standard library includes
#include <vector>

// Rcpp includes
#include <RcppArmadillo.h>

// Helper class to track contingency table for PY multiview sampler
struct ContingencyTracker {
    /* Class Members */
    // Dimensions
    size_t n_obs;
    double total_pairs;
    // State Tracking
    std::vector<arma::uvec> clus_allocs;        // [V] Current allocations
    std::vector<arma::uvec> clus_sizes;         // [V] Cluster counts
    std::vector<double> marginal_pair_counts;   // [V] Marginal pairs
    // Joint Tracking
    std::vector<std::vector<arma::umat>> joint_cluster_counts;  // [V][V] Intersection counts
    std::vector<std::vector<double>> joint_pair_counts;         // [V][V] Joint pairs
    
    /* Class methods */
    void init(int n_views, size_t n_data);
    void sync_view(size_t v, const arma::uvec& new_allocs);
    void apply_move(size_t v, int obs_i, int k_old, int k_new);
};