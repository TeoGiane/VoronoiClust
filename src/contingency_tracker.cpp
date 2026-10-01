#include "contingency_tracker.h"

void ContingencyTracker::init(size_t n_views, size_t n_data, size_t initial_cap) {
    // Initialize dimensions
    n_obs = n_data;
    total_pairs = (n_obs < 2) ? 0.0 : (static_cast<double>(n_obs) * (n_obs - 1.0)) / 2.0;
    // Start small; ensure_capacity() grows the K-indexed tables on demand.
    cap = std::max<size_t>(1, std::min(n_obs, initial_cap));
    // Set up trackers
    clus_allocs.assign(n_views, arma::zeros<arma::uvec>(n_obs));
    clus_sizes.assign(n_views, arma::zeros<arma::uvec>(cap));
    marginal_pair_counts.assign(n_views, 0.0);
    joint_cluster_counts.assign(n_views, std::vector<arma::umat>(n_views, arma::zeros<arma::umat>(cap, cap)));
    joint_pair_counts.assign(n_views, std::vector<double>(n_views, 0.0));
};

void ContingencyTracker::ensure_capacity(size_t needed) {
    if (needed <= cap) return;
    // Double, but never past n_obs: PYSampler guarantees cluster ids < n_data,
    // so that is a hard ceiling and the growth loop always terminates.
    const size_t new_cap = std::min(n_obs, std::max(needed, cap * 2));
    for (auto& s : clus_sizes) s.resize(new_cap);                 // preserves + zero-fills
    for (auto& row : joint_cluster_counts) {
        for (auto& m : row) m.resize(new_cap, new_cap);           // same
    }
    cap = new_cap;
};

void ContingencyTracker::sync_view(size_t v, const arma::uvec& new_allocs) {
    // Grow before indexing: this view's step may have created cluster ids
    // beyond the current capacity.
    if (new_allocs.n_elem > 0) ensure_capacity(static_cast<size_t>(new_allocs.max()) + 1);
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
    // k_new may be a brand-new cluster id beyond the current capacity.
    ensure_capacity(static_cast<size_t>(std::max(k_old, k_new)) + 1);
    // Fast marginal update.
    // NOTE the casts: clus_sizes is an arma::uvec, so without them the whole
    // right-hand side is evaluated in UNSIGNED arithmetic before it reaches
    // the double. Moving a point out of a size-10 cluster into a size-2 one
    // computes 2 - 9, which wraps to ~1.8e19 and destroys the counter for the
    // rest of the sweep. (The structurally identical expressions in
    // MultiViewPYSampler's gibbs_evaluator are accidentally safe, because a
    // leading double operand promotes before the subtraction happens.)
    const double n_new = static_cast<double>(clus_sizes[v](k_new));
    const double n_old = static_cast<double>(clus_sizes[v](k_old));
    marginal_pair_counts[v] += n_new - (n_old - 1.0);
    clus_sizes[v](k_old)--;
    clus_sizes[v](k_new)++;
    // Fast joint update
    for (size_t u = 0; u < clus_allocs.size(); ++u) {
        if (u == v) continue;
        int c = clus_allocs[u](obs_i);
        const double j_new = static_cast<double>(joint_cluster_counts[v][u](k_new, c));
        const double j_old = static_cast<double>(joint_cluster_counts[v][u](k_old, c));
        joint_pair_counts[v][u] += j_new - (j_old - 1.0);
        joint_pair_counts[u][v] = joint_pair_counts[v][u];
        joint_cluster_counts[v][u](k_old, c)--;
        joint_cluster_counts[v][u](k_new, c)++;
        joint_cluster_counts[u][v](c, k_old)--;
        joint_cluster_counts[u][v](c, k_new)++;
    }
    clus_allocs[v](obs_i) = k_new;
};