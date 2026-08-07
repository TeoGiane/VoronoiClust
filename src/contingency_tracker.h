#pragma once

// Standard library includes
#include <vector>
#include <algorithm>

// Rcpp includes
#include <RcppArmadillo.h>

// Helper class to track contingency table for PY multiview sampler
struct ContingencyTracker {
    /* Class Members */
    // Dimensions
    size_t n_obs;
    double total_pairs;
    // Current cluster-id capacity of every K-indexed structure below.
    // The joint tables are indexed by CLUSTER id, not by observation, so
    // sizing them n_obs x n_obs (their original shape) cost O(V^2 * n^2)
    // memory and -- far worse -- made sync_view's .zeros() calls O(n^2),
    // which dominated the entire sampler. They are grown on demand instead,
    // so there is no K_max to guess wrong.
    size_t cap = 0;
    // State Tracking
    std::vector<arma::uvec> clus_allocs;        // [V] Current allocations (length n_obs)
    std::vector<arma::uvec> clus_sizes;         // [V] Cluster counts (length cap)
    std::vector<double> marginal_pair_counts;   // [V] Marginal pairs
    // Joint Tracking
    std::vector<std::vector<arma::umat>> joint_cluster_counts;  // [V][V] Intersection counts (cap x cap)
    std::vector<std::vector<double>> joint_pair_counts;         // [V][V] Joint pairs
    
    /* Class methods */
    void init(size_t n_views, size_t n_data, size_t initial_cap = 16);
    // Grows every K-indexed structure so that ids up to `needed - 1` are
    // addressable. Armadillo's resize() preserves existing elements and
    // zero-fills the new tail, which is exactly the semantics wanted here.
    void ensure_capacity(size_t needed);
    void sync_view(size_t v, const arma::uvec& new_allocs);
    void apply_move(size_t v, int obs_i, int k_old, int k_new);

    // Range-safe reads. A brand-new cluster id gets SCORED by the driver's
    // gibbs_evaluator before anything has grown the tables for it, and that
    // path is const -- so out-of-range must read as zero rather than grow.
    // An id at or beyond `cap` has never been occupied, so 0 is correct.
    double size_of(size_t v, int k) const {
        return (k >= 0 && static_cast<size_t>(k) < cap)
             ? static_cast<double>(clus_sizes[v](k)) : 0.0;
    }
    double joint_of(size_t v, size_t u, int k, int c) const {
        return (k >= 0 && c >= 0 && static_cast<size_t>(k) < cap && static_cast<size_t>(c) < cap)
             ? static_cast<double>(joint_cluster_counts[v][u](k, c)) : 0.0;
    }
};