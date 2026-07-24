#include "contingency_tracker.h"

void ContingencyTracker::init(int n_views, size_t n_data) {
    // Initialize dimensions
    n_obs = n_data;
    total_pairs = (n_obs * (n_obs - 1)) / 2.0;
    // Set up trackers
    clus_allocs.assign(n_views, arma::zeros<arma::uvec>(n_obs));
    clus_sizes.assign(n_views, arma::zeros<arma::uvec>(n_obs));
    marginal_pair_counts.assign(n_views, 0.0);
    joint_cluster_counts.assign(n_views, std::vector<arma::umat>(n_views, arma::zeros<arma::umat>(n_obs, n_obs)));
    joint_pair_counts.assign(n_views, std::vector<double>(n_views, 0.0));
};

void ContingencyTracker::sync_view(size_t v, const arma::uvec& new_allocs) {
    // Get trackers in current view
    clus_allocs[v] = new_allocs;
    clus_sizes[v].zeros();
    marginal_pair_counts[v] = 0.0;
    // Rebuild marginals
    for (size_t i = 0; i < n_obs; ++i) {
        int k = clus_allocs[v](i);
        marginal_pair_counts[v] += clus_sizes[v](k); 
        clus_sizes[v](k)++;
    }
    // Rebuild joints
    for (size_t u = 0; u < clus_allocs.size(); ++u) {
        if (u == v) continue;
        joint_cluster_counts[v][u].zeros();
        joint_cluster_counts[u][v].zeros();
        joint_pair_counts[v][u] = 0.0;
        for (size_t i = 0; i < n_obs; ++i) {
            int k = clus_allocs[v](i);
            int c = clus_allocs[u](i);
            joint_pair_counts[v][u] += joint_cluster_counts[v][u](k, c);
            joint_cluster_counts[v][u](k, c)++;
            joint_cluster_counts[u][v](c, k)++; // Ensure symmetry
        }
        joint_pair_counts[u][v] = joint_pair_counts[v][u];
    }
    // Now the tracker is synced with the accepted clus_allocs in the given view
    return;
};

void ContingencyTracker::apply_move(size_t v, int obs_i, int k_old, int k_new) {
    // Do nothing if the cluster has not changed
    if (k_old == k_new) return;
    // Fast marginal update
    marginal_pair_counts[v] += clus_sizes[v](k_new) - (clus_sizes[v](k_old) - 1);
    clus_sizes[v](k_old)--;
    clus_sizes[v](k_new)++;
    // Fast joint update
    for (size_t u = 0; u < clus_allocs.size(); ++u) {
        if (u == v) continue;
        int c = clus_allocs[u](obs_i);
        joint_pair_counts[v][u] += joint_cluster_counts[v][u](k_new, c) - (joint_cluster_counts[v][u](k_old, c) - 1);
        joint_pair_counts[u][v] = joint_pair_counts[v][u];
        joint_cluster_counts[v][u](k_old, c)--;
        joint_cluster_counts[v][u](k_new, c)++;
        joint_cluster_counts[u][v](c, k_old)--;
        joint_cluster_counts[u][v](c, k_new)++;
    }
    clus_allocs[v](obs_i) = k_new;
};